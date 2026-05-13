"""
Baseline: DIN-SQL — dùng Vertex AI MaaS (Llama 4)
===================================================
Dùng lại GoogleGenAI wrapper có sẵn trong project.
Không cần Google AI Studio hay GOOGLE_API_KEY.

.env cần có (đã có sẵn):
    USE_VERTEX_AI=false          # MaaS tự xử lý, không cần đổi
    GOOGLE_CLOUD_PROJECT=text2sql-research
    VERTEX_AI_LOCATION=us-east5
    MODEL_NAME=meta/llama-4-maverick-17b-128e-instruct-maas

Pipeline 4 stages (DIN-SQL NeurIPS 2023):
  Stage 1 – Schema Linking   : tìm bảng/cột liên quan
  Stage 2 – Classification   : EASY / NON-NESTED / NESTED
  Stage 3 – SQL Generation   : prompt khác nhau theo loại
  Stage 4 – Self-Correction  : sửa lỗi nếu SQL không chạy được

--wikisql_fast  : Bỏ Stage 1 & 2 (WikiSQL luôn single-table = EASY).
                  Giảm từ 3-4 calls/q xuống 1-2 calls/q.
                  Dùng khi chạy full WikiSQL dev set (8421 mẫu).

Usage:
  # Spider (full pipeline)
  python scripts/baselines/run_dinsql.py \
      --questions data/raw/spider/dev.json \
      --db        data/raw/spider/database \
      --output    results/predictions_dinsql_spider.tsv

  # WikiSQL (fast mode — khuyến nghị cho full run)
  python scripts/baselines/run_dinsql.py \
      --questions data/raw/wikisql/dev_spider_format.json \
      --db        data/raw/wikisql/database \
      --output    results/predictions_dinsql_wikisql.tsv \
      --wikisql_fast \
      --delay 0.5 \
      --resume
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
import platform

DEFAULT_DELAY = 1.5   # giây giữa mỗi API call (có thể override qua --delay)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.models.google_genai import GoogleGenAI


# ── Build model ───────────────────────────────────────────────────────────────

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


# delay được set từ args sau khi parse, dùng biến module-level
_INTER_CALL_DELAY = DEFAULT_DELAY

def _generate(model: GoogleGenAI, prompt: str) -> str:
    """Gọi model, trả về text. Retry 1 lần nếu gặp lỗi thoáng qua."""
    time.sleep(_INTER_CALL_DELAY)
    for attempt in range(2):
        try:
            return model.generate(prompt, temperature=0.0) or ""
        except Exception as e:
            err_str = str(e)
            # 429 Rate limit: back off thêm 5s rồi thử lại 1 lần
            if "429" in err_str and attempt == 0:
                logger.warning("429 rate limit — backing off 5s")
                time.sleep(5)
                continue
            # 5xx / 401: không retry, trả về rỗng để caller xử lý
            logger.error(f"Generation error (attempt {attempt+1}): {e}")
            if attempt == 1:
                return ""
    return ""


# ── Schema helpers ────────────────────────────────────────────────────────────

def _load_schema(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    columns = {}
    lines   = []
    for table in tables:
        cur.execute(f"PRAGMA table_info('{table}');")
        cols = cur.fetchall()
        col_defs = [(c[1], c[2]) for c in cols]
        columns[table] = [c[0] for c in col_defs]
        col_str = ", ".join(f"{n} {t}" for n, t in col_defs)
        lines.append(f"Table {table}: ({col_str})")
    fks = []
    for table in tables:
        cur.execute(f"PRAGMA foreign_key_list('{table}');")
        for row in cur.fetchall():
            fks.append((table, row[3], row[2], row[4]))
    if fks:
        lines.append("Foreign Keys:")
        for src_t, src_c, ref_t, ref_c in fks:
            lines.append(f"  {src_t}.{src_c} = {ref_t}.{ref_c}")
    conn.close()
    return {"tables": tables, "columns": columns, "foreign_keys": fks,
            "schema_str": "\n".join(lines)}


def _try_execute(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA query_only = ON;")
        conn.execute(sql)
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)


def _extract_sql(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"```sql", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    m = re.search(r"(SELECT\b.*?)(?:;|\Z)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        return " ".join(m.group(1).strip().rstrip(";").split())
    return ""


# ── Stage 1 – Schema Linking ──────────────────────────────────────────────────

SCHEMA_LINK_PROMPT = """\
You are a database expert. Identify which tables and columns are needed \
to answer the question. Be concise and precise.

### Database Schema
{schema}

### Question
{question}

Answer ONLY in this exact format (no extra text):
Tables: <comma-separated table names>
Columns: <comma-separated table.column pairs>
Foreign keys used: <comma-separated t1.col=t2.col pairs, or NONE>
"""

def stage1_schema_link(model, schema_info, question):
    raw    = _generate(model, SCHEMA_LINK_PROMPT.format(
        schema=schema_info["schema_str"], question=question))
    result = {"tables": [], "columns": [], "foreign_keys": []}
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("tables:"):
            result["tables"]  = [t.strip() for t in line[7:].split(",")  if t.strip()]
        elif line.lower().startswith("columns:"):
            result["columns"] = [c.strip() for c in line[8:].split(",")  if c.strip()]
        elif line.lower().startswith("foreign keys used:"):
            fk_str = line[18:].strip()
            if fk_str.upper() != "NONE":
                result["foreign_keys"] = [f.strip() for f in fk_str.split(",") if f.strip()]
    return result


# ── Stage 2 – Classification ──────────────────────────────────────────────────

CLASSIFY_PROMPT = """\
Classify the SQL query needed to answer this question into one category.

Categories:
- EASY: single table, no JOIN, no subquery, no GROUP BY
- NON-NESTED: may use JOIN, GROUP BY, HAVING, ORDER BY — but NO subquery
- NESTED: requires subqueries (IN, NOT IN, EXISTS, correlated)

Tables: {tables}
Columns: {columns}
Foreign keys: {fks}
Question: {question}

Reply with ONLY one word: EASY, NON-NESTED, or NESTED.
"""

def stage2_classify(model, link_result, question):
    raw = _generate(model, CLASSIFY_PROMPT.format(
        tables  = ", ".join(link_result["tables"])  or "unknown",
        columns = ", ".join(link_result["columns"]) or "unknown",
        fks     = ", ".join(link_result["foreign_keys"]) or "NONE",
        question=question,
    )).strip().upper()
    for cls in ("NESTED", "NON-NESTED", "EASY"):
        if cls in raw:
            return cls
    return "NON-NESTED"


# ── Stage 3 – SQL Generation ──────────────────────────────────────────────────

_GEN_BASE = """\
You are a SQL expert. Write a single SQL SELECT statement that answers the question.
Output ONLY the SQL — no explanation, no markdown, no code fences.

### Database Schema
{schema}

Relevant tables : {tables}
Relevant columns: {columns}
Foreign keys    : {fks}

### Question
{question}

### Hint
{hint}

### SQL
SELECT"""

_HINTS = {
    "EASY":        "Simple single-table query. No JOINs or subqueries needed.",
    "NON-NESTED":  "May need JOINs, GROUP BY, HAVING, ORDER BY — but NO subqueries.",
    "NESTED":      "Likely needs nested subqueries (IN, NOT IN, EXISTS). Think step by step.",
}

def stage3_generate(model, schema_info, link_result, query_class, question):
    tables  = ", ".join(link_result["tables"])  or schema_info["schema_str"][:120]
    columns = ", ".join(link_result["columns"]) or "all"
    fks     = ", ".join(link_result["foreign_keys"]) or "NONE"
    prompt  = _GEN_BASE.format(
        schema=schema_info["schema_str"], tables=tables,
        columns=columns, fks=fks, question=question,
        hint=_HINTS.get(query_class, _HINTS["NON-NESTED"]),
    )
    raw = _generate(model, prompt)
    if raw.strip().upper().startswith("SELECT"):
        sql = _extract_sql(raw)
    else:
        sql = _extract_sql("SELECT " + raw)
    return sql or "SELECT 1"


# ── Stage 4 – Self-Correction ─────────────────────────────────────────────────

SELFCORRECT_PROMPT = """\
You are a SQL debugger. Fix the SQL query so it executes correctly.
Output ONLY the corrected SQL — no explanation, no markdown.

### Database Schema
{schema}

### Question
{question}

### Incorrect SQL
{sql}

### SQLite Error
{error}

### Corrected SQL
SELECT"""

def stage4_selfcorrect(model, schema_info, question, sql, error):
    raw = _generate(model, SELFCORRECT_PROMPT.format(
        schema=schema_info["schema_str"],
        question=question, sql=sql, error=error,
    ))
    if raw.strip().upper().startswith("SELECT"):
        corrected = _extract_sql(raw)
    else:
        corrected = _extract_sql("SELECT " + raw)
    return corrected if corrected else sql


# ── WikiSQL fast: Stage 1+2 được bypass ───────────────────────────────────────
# WikiSQL luôn là single-table, không có JOIN, không có subquery.
# → Stage 1 (schema linking) không cần thiết: chỉ có 1 table.
# → Stage 2 (classification) luôn trả về EASY.
# → Tiết kiệm 2 API calls mỗi câu hỏi.

def _wikisql_fast_link(schema_info: dict) -> dict:
    """Trả về link_result bao gồm tất cả bảng/cột — không gọi LLM."""
    all_cols = []
    for table, cols in schema_info["columns"].items():
        for col in cols:
            all_cols.append(f"{table}.{col}")
    return {
        "tables": schema_info["tables"],
        "columns": all_cols,
        "foreign_keys": [],
    }


# ── Full DIN-SQL pipeline ─────────────────────────────────────────────────────

def run_dinsql(model, db_path, question,
               skip_selfcorrect=False,
               wikisql_fast=False):
    schema_info = _load_schema(db_path)

    if wikisql_fast:
        # Bypass Stage 1 & 2 — WikiSQL is always single-table EASY
        link_result = _wikisql_fast_link(schema_info)
        query_class = "EASY"
    else:
        link_result = stage1_schema_link(model, schema_info, question)
        query_class = stage2_classify(model, link_result, question)

    sql = stage3_generate(model, schema_info, link_result, query_class, question)

    self_corrected = False
    exec_error     = ""
    if not skip_selfcorrect and sql != "SELECT 1":
        ok, err = _try_execute(db_path, sql)
        if not ok:
            exec_error = err
            corrected  = stage4_selfcorrect(model, schema_info, question, sql, err)
            if corrected and corrected != sql:
                sql = corrected
                self_corrected = True

    return {"sql": sql, "query_class": query_class,
            "self_corrected": self_corrected, "error": exec_error}


# ── Resume helper ─────────────────────────────────────────────────────────────

def _start_caffeinate():
    """
    Ngăn máy ngủ trong suốt quá trình chạy.
    - macOS : caffeinate -i  (prevent idle sleep)
    - Linux : systemd-inhibit hoặc skip nếu không có
    Trả về process object (cần .terminate() khi xong) hoặc None.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            proc = subprocess.Popen(
                ["caffeinate", "-i"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("☕ caffeinate started (macOS) — máy sẽ không ngủ.")
            return proc
        elif system == "Linux":
            # systemd-inhibit giữ wake lock tương tự caffeinate
            proc = subprocess.Popen(
                ["systemd-inhibit", "--what=idle:sleep", "--who=run_dinsql",
                 "--why=Long-running LLM job", "--mode=block",
                 "sleep", "infinity"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("☕ systemd-inhibit started (Linux) — máy sẽ không ngủ.")
            return proc
        else:
            logger.warning(f"caffeinate: unsupported OS ({system}), skipping.")
            return None
    except FileNotFoundError:
        logger.warning("caffeinate / systemd-inhibit không tìm thấy — bỏ qua.")
        return None
    except Exception as e:
        logger.warning(f"caffeinate failed to start: {e}")
        return None


def _count_existing_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DIN-SQL baseline — Vertex AI MaaS (Llama 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--db",        required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--limit",           type=int,   default=None)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--checkpoint_size", type=int,   default=None)
    parser.add_argument("--skip_selfcorrect", action="store_true",
                        help="Disable Stage 4 self-correction")
    parser.add_argument("--wikisql_fast", action="store_true",
                        help=(
                            "WikiSQL fast mode: skip Stage 1 (schema linking) và "
                            "Stage 2 (classification). WikiSQL luôn là single-table EASY, "
                            "nên 2 stages này lãng phí 2 API calls/câu. "
                            "Giảm từ 3-4 calls/q xuống 1-2 calls/q."
                        ))
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Giây delay giữa mỗi API call (default: {DEFAULT_DELAY}). "
                             "Giảm xuống 0.5 nếu ít bị 429.")
    args = parser.parse_args()

    # Ngăn máy ngủ trong suốt quá trình chạy
    _caff = _start_caffeinate()

    try:
        _run(args)
    finally:
        if _caff is not None:
            _caff.terminate()
            logger.info("☕ caffeinate stopped.")


def _run(args):
    """Logic chính tách ra để caffeinate có thể wrap bằng try/finally."""
    global _INTER_CALL_DELAY
    _INTER_CALL_DELAY = args.delay

    # Load data
    with open(args.questions, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    logger.info(f"Loaded {len(data)} questions")

    # Resume
    already_done = 0
    if args.resume and os.path.exists(args.output):
        already_done = _count_existing_lines(args.output)
        logger.info(f"Resuming — skipping {already_done} done")

    remaining = data[already_done:]
    if not remaining:
        logger.info("Nothing left to do.")
        return

    # Build model
    model = _build_model()
    model_name = os.getenv("MODEL_NAME") or "meta/llama-4-maverick-17b-128e-instruct-maas"

    if args.wikisql_fast:
        calls_per_q = "1-2 (fast mode: Stage 1+2 bypassed)"
    elif args.skip_selfcorrect:
        calls_per_q = "3 (no self-correction)"
    else:
        calls_per_q = "3-4"

    logger.info(f"Model      : {model_name}")
    logger.info(f"Calls/q    : {calls_per_q}")
    logger.info(f"Delay/call : {_INTER_CALL_DELAY}s")
    logger.info(f"Self-corr  : {'DISABLED' if args.skip_selfcorrect else 'ENABLED'}")
    logger.info(f"WikiSQL fast: {'YES — Stage 1+2 skipped' if args.wikisql_fast else 'NO'}")

    # Ước tính thời gian
    n_remaining = len(remaining)
    if args.wikisql_fast:
        avg_calls = 1.5  # trung bình 1-2
    else:
        avg_calls = 3.5
    est_secs = n_remaining * (avg_calls * _INTER_CALL_DELAY + avg_calls * 1.5)  # ~1.5s latency/call
    est_hours = est_secs / 3600
    logger.info(f"Estimated  : ~{est_hours:.1f}h for {n_remaining} remaining questions")

    stats = {"EASY": 0, "NON-NESTED": 0, "NESTED": 0,
             "corrected": 0, "failed": 0}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    new_count = 0

    mode = "a" if (args.resume and already_done > 0) else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        pbar = tqdm(remaining, desc="[DIN-SQL]", unit="q")
        for item in pbar:
            question = item.get("question", "")
            db_id    = item.get("db_id", "unknown")
            db_path  = os.path.join(args.db, db_id, f"{db_id}.sqlite")

            if not os.path.exists(db_path):
                logger.warning(f"DB not found: {db_path}")
                sql = "SELECT 1"
                stats["failed"] += 1
            else:
                try:
                    result = run_dinsql(
                        model, db_path, question,
                        skip_selfcorrect=args.skip_selfcorrect,
                        wikisql_fast=args.wikisql_fast,
                    )
                    sql = result["sql"]
                    stats[result["query_class"]] = stats.get(result["query_class"], 0) + 1
                    if result["self_corrected"]:
                        stats["corrected"] += 1
                except Exception as e:
                    logger.error(f"DIN-SQL failed on '{question[:50]}': {e}")
                    sql = "SELECT 1"
                    stats["failed"] += 1

            out_f.write(f"{sql.replace(chr(10), ' ').strip()}\t{db_id}\n")
            out_f.flush()
            new_count += 1

            pbar.set_postfix({
                "E": stats["EASY"], "NN": stats["NON-NESTED"],
                "N": stats["NESTED"], "fix": stats["corrected"],
            })

            if args.checkpoint_size and new_count >= args.checkpoint_size:
                logger.info(f"\n✓ Checkpoint {new_count}. Re-run with --resume.")
                break

    total = already_done + new_count
    logger.info(f"\n✓ Done. Total={total} | New={new_count}")
    logger.info(f"  EASY={stats['EASY']} | NON-NESTED={stats['NON-NESTED']} | "
                f"NESTED={stats['NESTED']} | corrected={stats['corrected']} | "
                f"failed={stats['failed']}")
    logger.info(f"Output → {args.output}")
    logger.info("\nEvaluate:")
    logger.info(f"  python scripts/evaluate_spider.py \\")
    logger.info(f"      --gold data/raw/spider/dev.json --db data/raw/spider/database \\")
    logger.info(f"      --predict {args.output} --etype all")


if __name__ == "__main__":
    main()