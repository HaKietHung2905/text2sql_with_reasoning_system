"""
Baseline: ChatGPT Zero-shot (Standalone)
=========================================

Backend: Vertex AI MaaS (Llama 4) — dùng lại .env hiện có.

Output: TSV  →  <sql>\t<db_id>

Usage (Spider):
  python scripts/baselines/run_zeroshot_gpt.py \
      --questions data/raw/spider/dev.json \
      --db        data/raw/spider/database \
      --output    results/predictions_zeroshot_spider.tsv \
      --limit     5

Usage (WikiSQL):
  python scripts/baselines/run_zeroshot_gpt.py \
      --questions data/raw/wikisql/dev_spider_format.json \
      --db        data/raw/wikisql/database \
      --output    results/predictions_zeroshot_wikisql.tsv \
      --limit     5

Evaluate:
  python scripts/evaluate_spider.py \
      --gold    data/raw/spider/dev.json \
      --db      data/raw/spider/database \
      --predict results/predictions_zeroshot_spider.tsv \
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
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import time 
import subprocess
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
    """Khởi tạo model từ .env — ưu tiên MODEL_NAME (Llama 4 MaaS)."""
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
# Schema helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_schema_string(db_path: str) -> str:
    """Đọc schema từ SQLite, trả về chuỗi dạng:
       Table singer: (singer_id INTEGER, name TEXT, ...)
    """
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
    # Foreign keys
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
# Zero-shot prompt  (1 lần gọi duy nhất — không có few-shot, không có chain)
# ─────────────────────────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """\
You are a SQL expert. Given the database schema below, write a single \
SQL SELECT statement that answers the question.

Rules:
- Output ONLY the SQL query.
- No explanation, no markdown, no code fences.
- Use only tables and columns that exist in the schema.
- Start your answer with SELECT.

### Database Schema
{schema}

### Question
{question}

### SQL
SELECT"""


# ─────────────────────────────────────────────────────────────────────────────
# SQL extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sql(raw: str) -> str:
    if not raw:
        return "SELECT 1"
    # Strip think blocks (CoT models)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip markdown fences
    raw = re.sub(r"```sql", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()

    # Model thường trả về phần sau "SELECT" (vì prompt pre-fill là "SELECT")
    # → ghép lại
    first = raw.splitlines()[0].strip() if raw else ""
    if first and not first.upper().startswith("SELECT"):
        raw = "SELECT " + raw

    m = re.search(r"(SELECT\b.*?)(?:;|\Z)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip().rstrip(";")
        # Bỏ prose sau blank line (CoT suffix)
        sql = sql.split("\n\n")[0]
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
            description="Zero-shot baseline — 1 LLM call per question, no extras",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
        parser.add_argument("--questions", required=True,
                            help="Path to Spider/WikiSQL dev JSON")
        parser.add_argument("--db", required=True,
                            help="Database directory (<db_id>/<db_id>.sqlite)")
        parser.add_argument("--output", required=True,
                            help="Output TSV file path")
        parser.add_argument("--limit", type=int, default=None,
                            help="Max questions to process (for testing)")
        parser.add_argument("--resume", action="store_true",
                            help="Skip already-done lines in output file")
        parser.add_argument("--checkpoint_size", type=int, default=None,
                            help="Stop after N new predictions (staged runs)")
        args = parser.parse_args()

        # ── Load questions ──────────────────────────────────────────────────────
        with open(args.questions, "r", encoding="utf-8") as f:
            data = json.load(f)
        if args.limit:
            data = data[: args.limit]
        logger.info(f"Loaded {len(data)} questions from {args.questions}")

        # ── Resume ──────────────────────────────────────────────────────────────
        already_done = 0
        if args.resume and os.path.exists(args.output):
            already_done = _count_existing_lines(args.output)
            logger.info(f"Resuming — skipping {already_done} already-done")

        remaining = data[already_done:]
        if not remaining:
            logger.info("Nothing left to do.")
            return

        # ── Model ────────────────────────────────────────────────────────────────
        model      = _build_model()
        model_name = os.getenv("MODEL_NAME") or "meta/llama-4-maverick-17b-128e-instruct-maas"
        logger.info(f"Model  : {model_name}")
        logger.info(f"Method : Zero-shot (1 call/question)")
        logger.info(f"Extras : None (no ReasoningBank, no ChromaDB, no Semantic RAG)")

        # ── Generate ─────────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        new_count = 0
        failed    = 0

        mode = "a" if (args.resume and already_done > 0) else "w"
        with open(args.output, mode, encoding="utf-8") as out_f:
            for item in tqdm(remaining, desc="[Zero-shot]", unit="q"):
                question = item.get("question", "")
                db_id    = item.get("db_id", "unknown")
                db_path  = os.path.join(args.db, db_id, f"{db_id}.sqlite")

                if not os.path.exists(db_path):
                    logger.warning(f"DB not found: {db_path}")
                    sql = "SELECT 1"
                    failed += 1
                else:
                    try:
                        schema = _get_schema_string(db_path)
                        prompt = ZERO_SHOT_PROMPT.format(schema=schema, question=question)
                        time.sleep(INTER_CALL_DELAY)
                        raw    = model.generate(prompt, temperature=0.0)
                        sql    = _extract_sql(raw)
                    except Exception as e:
                        logger.error(f"[{already_done + new_count}] Failed: {e}")
                        sql = "SELECT 1"
                        failed += 1

                out_f.write(f"{sql.replace(chr(10), ' ').strip()}\t{db_id}\n")
                out_f.flush()
                new_count += 1

                if args.checkpoint_size and new_count >= args.checkpoint_size:
                    logger.info(
                        f"\n✓ Checkpoint: {new_count} new predictions "
                        f"({already_done + new_count} total). Re-run with --resume."
                    )
                    break

        total = already_done + new_count
        logger.info(f"\n✓ Done. Total={total} | New={new_count} | Failed/fallback={failed}")
        logger.info(f"Output → {args.output}")
        logger.info("\nNext — evaluate:")
        logger.info(f"  python scripts/evaluate_spider.py \\")
        logger.info(f"      --gold data/raw/spider/dev.json \\")
        logger.info(f"      --db   data/raw/spider/database \\")
        logger.info(f"      --predict {args.output} --etype all")
    finally:
        caffeinate.terminate()

if __name__ == "__main__":
    main()