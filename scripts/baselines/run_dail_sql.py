"""
Baseline: DAIL-SQL (VLDB 2024) — Standalone Script
====================================================
Re-implements core DAIL-SQL pipeline:
  "DAIL-SQL: Efficient Prompt Engineering for Large Language Models in
   Text-to-SQL" — Gao et al., VLDB 2024.

Key ideas:
  1. Few-shot retrieval  : tìm k câu hỏi training tương tự nhất
                           bằng embedding cosine similarity
  2. SQL skeleton masking: che giá trị literal trong SQL mẫu
                           → model học cấu trúc, không học giá trị
  3. Prompt assembly     : schema + k examples (Q+SQL) + test question
  4. Generation          : 1 LLM call với full few-shot context

Backend: Vertex AI MaaS (Llama 4) — dùng lại .env hiện có.

Usage (Spider):
  python scripts/baselines/run_dail_sql.py \
      --questions data/raw/spider/dev.json \
      --db        data/raw/spider/database \
      --train     data/raw/spider/train_spider.json \
      --output    results/predictions_dail_sql_spider.tsv \
      --shots     3 \
      --limit     5

Usage (WikiSQL):
  python scripts/baselines/run_dail_sql.py \
      --questions data/raw/wikisql/dev_spider_format.json \
      --db        data/raw/wikisql/database \
      --train     data/raw/wikisql/train_spider_format.json \
      --output    results/predictions_dail_sql_wikisql.tsv \
      --shots     3 \
      --limit     5

Evaluate:
  python scripts/evaluate_spider.py \
      --gold    data/raw/spider/dev.json \
      --db      data/raw/spider/database \
      --predict results/predictions_dail_sql_spider.tsv \
      --etype   all
"""

import sys
import os
import json
import argparse
import re
import sqlite3
import logging
import warnings
import numpy as np
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.models.google_genai import GoogleGenAI
INTER_CALL_DELAY = 1.0 

# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_model() -> GoogleGenAI:
    model_name = (
        os.getenv("MODEL_NAME")
        or os.getenv("GEMINI_MODEL")
        or "meta/llama-4-maverick-17b-128e-instruct-maas"
    )
    return GoogleGenAI(
        model_name=model_name,
        use_vertex_ai=True,
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("VERTEX_AI_LOCATION", "us-east5"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + retrieval
# ─────────────────────────────────────────────────────────────────────────────

_EMBED_MODEL = None

def _get_embedder():
    """Lazy-load sentence-transformers (đã có trong requirements.txt)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (all-MiniLM-L6-v2)…")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model ready.")
    return _EMBED_MODEL


def _embed_batch(texts: List[str]) -> np.ndarray:
    """Trả về ma trận (N, D) đã normalize."""
    embedder = _get_embedder()
    vecs = embedder.encode(texts, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True)
    # L2 normalize để cosine = dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vecs / norms


def _embed_single(text: str) -> np.ndarray:
    return _embed_batch([text])[0]


class ExampleIndex:
    """
    Index training examples để tìm k-nearest neighbours nhanh.
    Dùng cosine similarity trên question embeddings.
    """

    def __init__(self, train_data: List[Dict]):
        logger.info(f"Building DAIL-SQL example index ({len(train_data)} examples)…")
        self.examples = train_data
        questions = [ex.get("question", "") for ex in train_data]
        self.vectors = _embed_batch(questions)   # (N, D)
        logger.info("Index ready.")

    def retrieve(self, question: str, k: int = 3,
                 db_id: str = None) -> List[Dict]:
        """
        Trả về k training examples tương tự nhất.
        Nếu db_id được cung cấp → ưu tiên cùng database (DAIL-SQL paper).
        """
        q_vec = _embed_single(question)                     # (D,)
        scores = self.vectors @ q_vec                       # (N,) cosine sim

        # Mask example có question giống hệt (tránh leak dev→train)
        for i, ex in enumerate(self.examples):
            if ex.get("question", "").strip() == question.strip():
                scores[i] = -999.0

        # Lấy top-k * 3 để re-rank theo db_id
        top_idx = np.argsort(scores)[::-1][: k * 3]
        candidates = [(self.examples[i], scores[i]) for i in top_idx]

        # Ưu tiên same-db examples (DAIL-SQL heuristic)
        if db_id:
            same_db = [(ex, s) for ex, s in candidates if ex.get("db_id") == db_id]
            diff_db = [(ex, s) for ex, s in candidates if ex.get("db_id") != db_id]
            candidates = same_db + diff_db

        return [ex for ex, _ in candidates[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# SQL skeleton masking  (core DAIL-SQL contribution)
# ─────────────────────────────────────────────────────────────────────────────

def _make_sql_skeleton(sql: str) -> str:
    """
    Che các giá trị literal trong SQL → model học cấu trúc.
    VD:  WHERE age > 30 AND name = 'Taylor'
    →    WHERE age > _ AND name = _
    """
    if not sql:
        return sql
    # String literals  'value'  →  _
    sql = re.sub(r"'[^']*'", "_", sql)
    # Numeric literals  42 / 3.14  →  _
    sql = re.sub(r"\b\d+(\.\d+)?\b", "_", sql)
    # Collapse multiple spaces
    return " ".join(sql.split())


# ─────────────────────────────────────────────────────────────────────────────
# Schema helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_schema_string(db_path: str) -> str:
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cursor.fetchall()]
    lines  = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        cols    = cursor.fetchall()
        col_str = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        lines.append(f"Table {table}: ({col_str})")
    fk_lines = []
    for table in tables:
        cursor.execute(f"PRAGMA foreign_key_list('{table}');")
        for row in cursor.fetchall():
            fk_lines.append(f"  {table}.{row[3]} = {row[2]}.{row[4]}")
    if fk_lines:
        lines.append("Foreign Keys:")
        lines.extend(fk_lines)
    conn.close()
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DAIL-SQL prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_dail_prompt(schema: str, examples: List[Dict],
                       question: str, use_skeleton: bool = True) -> str:
    """
    Xây dựng prompt theo format DAIL-SQL:
      schema + k×(Q + SQL skeleton) + test question
    """
    lines = [
        "You are a SQL expert. Given the database schema and example "
        "question-SQL pairs below, write a SQL SELECT statement for the "
        "new question.",
        "",
        "Rules:",
        "- Output ONLY the SQL query.",
        "- No explanation, no markdown, no code fences.",
        "- Use only tables/columns that exist in the schema.",
        "",
        "### Database Schema",
        schema,
        "",
    ]

    if examples:
        lines.append("### Examples")
        for i, ex in enumerate(examples, 1):
            q   = ex.get("question", "")
            sql = ex.get("query", ex.get("sql", ""))
            if use_skeleton:
                sql = _make_sql_skeleton(sql)
            lines.append(f"-- Q{i}: {q}")
            lines.append(f"   {sql}")
            lines.append("")

    lines += [
        "### New Question",
        f"-- Q: {question}",
        "",
        "### SQL",
        "SELECT",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SQL extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sql(raw: str) -> str:
    if not raw:
        return "SELECT 1"
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```sql", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()
    first = raw.splitlines()[0].strip() if raw else ""
    if first and not first.upper().startswith("SELECT"):
        raw = "SELECT " + raw
    m = re.search(r"(SELECT\b.*?)(?:;|\Z)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip().rstrip(";").split("\n\n")[0]
        return " ".join(sql.split())
    return "SELECT 1"


# ─────────────────────────────────────────────────────────────────────────────
# Resume helper
# ─────────────────────────────────────────────────────────────────────────────

def _count_existing_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        parser = argparse.ArgumentParser(
            description="DAIL-SQL baseline — few-shot retrieval + SQL skeleton",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument("--questions", required=True,
                            help="Dev JSON (Spider/WikiSQL)")
        parser.add_argument("--db", required=True,
                            help="Database directory")
        parser.add_argument("--train", required=True,
                            help="Training JSON for few-shot retrieval "
                                "(e.g. data/raw/spider/train_spider.json)")
        parser.add_argument("--output", required=True,
                            help="Output TSV file")
        parser.add_argument("--shots", type=int, default=3,
                            help="Number of few-shot examples (default: 3)")
        parser.add_argument("--no_skeleton", action="store_true",
                            help="Disable SQL skeleton masking (dùng full SQL trong examples)")
        parser.add_argument("--limit",           type=int, default=None)
        parser.add_argument("--resume",          action="store_true")
        parser.add_argument("--checkpoint_size", type=int, default=None)
        args = parser.parse_args()

        # ── Load training data → build index ───────────────────────────────────
        logger.info(f"Loading training examples from {args.train}…")
        with open(args.train, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        logger.info(f"  {len(train_data)} training examples loaded")

        index = ExampleIndex(train_data)

        # ── Load dev questions ──────────────────────────────────────────────────
        with open(args.questions, "r", encoding="utf-8") as f:
            data = json.load(f)
        if args.limit:
            data = data[: args.limit]
        logger.info(f"Loaded {len(data)} questions from {args.questions}")

        # ── Resume ──────────────────────────────────────────────────────────────
        already_done = 0
        if args.resume and os.path.exists(args.output):
            already_done = _count_existing_lines(args.output)
            logger.info(f"Resuming — skipping {already_done} done")

        remaining = data[already_done:]
        if not remaining:
            logger.info("Nothing left to do.")
            return

        # ── Build model ──────────────────────────────────────────────────────────
        model      = _build_model()
        model_name = os.getenv("MODEL_NAME") or "meta/llama-4-maverick-17b-128e-instruct-maas"
        use_skeleton = not args.no_skeleton

        logger.info(f"Model   : {model_name}")
        logger.info(f"Method  : DAIL-SQL | shots={args.shots} | "
                    f"skeleton={'ON' if use_skeleton else 'OFF'}")
        logger.info(f"API     : Vertex AI MaaS | 1 call/question")

        # ── Generate ─────────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        new_count = 0
        failed    = 0

        mode = "a" if (args.resume and already_done > 0) else "w"
        with open(args.output, mode, encoding="utf-8") as out_f:
            for item in tqdm(remaining, desc="[DAIL-SQL]", unit="q"):
                question = item.get("question", "")
                db_id    = item.get("db_id", "unknown")
                db_path  = os.path.join(args.db, db_id, f"{db_id}.sqlite")

                if not os.path.exists(db_path):
                    logger.warning(f"DB not found: {db_path}")
                    sql = "SELECT 1"
                    failed += 1
                else:
                    try:
                        # Stage 1: retrieve similar examples
                        examples = index.retrieve(question, k=args.shots, db_id=db_id)

                        # Stage 2: build DAIL-SQL prompt
                        schema = _get_schema_string(db_path)
                        prompt = _build_dail_prompt(schema, examples, question,
                                                use_skeleton=use_skeleton)

                        # Stage 3: generate (1 LLM call)
                        raw = model.generate(prompt, temperature=0.0)
                        sql = _extract_sql(raw)

                    except Exception as e:
                        logger.error(f"[{already_done + new_count}] Failed: {e}")
                        sql = "SELECT 1"
                        failed += 1

                out_f.write(f"{sql.replace(chr(10), ' ').strip()}\t{db_id}\n")
                out_f.flush()
                new_count += 1

                if args.checkpoint_size and new_count >= args.checkpoint_size:
                    logger.info(
                        f"\n✓ Checkpoint: {new_count} new "
                        f"({already_done + new_count} total). Re-run with --resume."
                    )
                    break

        total = already_done + new_count
        logger.info(f"\n✓ Done. Total={total} | New={new_count} | Failed={failed}")
        logger.info(f"Output → {args.output}")
        logger.info("\nEvaluate:")
        logger.info(f"  python scripts/evaluate_spider.py \\")
        logger.info(f"      --gold data/raw/spider/dev.json \\")
        logger.info(f"      --db   data/raw/spider/database \\")
        logger.info(f"      --predict {args.output} --etype all")
    finally:
        caffeinate.terminate()

if __name__ == "__main__":
    main()