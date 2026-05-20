#!/usr/bin/env python3
"""
regen_failed.py — Re-generate failed predictions in-place
==========================================================

MODE 1 — Fix SELECT 1 fallbacks (default, auto-detects):
    python scripts/baselines/regen_failed.py \\
        --predict   results/predictions_wikisql_v2.tsv \\
        --questions data/raw/wikisql/dev_spider_format.json \\
        --db        data/raw/wikisql/database \\
        --use_reasoning_bank --use_chromadb --use_semantic

MODE 2 — Few-shot regeneration using same-table gold examples (BEST for EX):
    # Step 1: generate fresh failures CSV
    python scripts/evaluate_wikisql.py \\
        --gold    data/raw/wikisql/dev_spider_format.json \\
        --table   data/raw/wikisql/tables.json \\
        --predict results/predictions_wikisql_v2.tsv \\
        --etype   all --save_em_failures results/em_failures.csv

    # Step 2: regenerate failures using same-table few-shot
    python scripts/baselines/regen_failed.py \\
        --predict          results/predictions_wikisql_v2.tsv \\
        --questions        data/raw/wikisql/dev_spider_format.json \\
        --db               data/raw/wikisql/database \\
        --few_shot_mode \\
        --gold_file        data/raw/wikisql/dev_spider_format.json \\
        --from_failures_csv results/em_failures.csv

MODE 3 — Force COUNT for "how many" bare-SELECT predictions:
    python scripts/baselines/regen_failed.py \\
        --predict   results/predictions_wikisql_v2.tsv \\
        --questions data/raw/wikisql/dev_spider_format.json \\
        --db        data/raw/wikisql/database \\
        --force_count

MODE 4 — Specific line numbers:
    python scripts/baselines/regen_failed.py \\
        --predict results/predictions_wikisql_v2.tsv \\
        --questions data/raw/wikisql/dev_spider_format.json \\
        --db data/raw/wikisql/database \\
        --lines 11 12 13 47 502

MODE 5 — Interactive:
    python scripts/baselines/regen_failed.py ... --interactive
"""

import sys, os, json, shutil, argparse, logging, warnings, time, re, subprocess, csv
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve()
for _ in range(5):
    _PROJECT_ROOT = _PROJECT_ROOT.parent
    if (_PROJECT_ROOT / 'src').exists() or (_PROJECT_ROOT / 'configs').exists():
        break
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        try:
            import yaml; return yaml.safe_load(f)
        except ImportError:
            return {}


# ─── TSV helpers ──────────────────────────────────────────────────────────────

def load_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def save_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")

def backup_tsv(path):
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak

def find_failed_lines(rows, marker="SELECT 1"):
    return [i+1 for i, row in enumerate(rows)
            if row.split("\t")[0].strip().upper() == marker.upper()]


# ─── Auth ─────────────────────────────────────────────────────────────────────

def get_access_token():
    r = subprocess.run(["gcloud", "auth", "print-access-token"],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError(f"gcloud auth failed: {r.stderr.strip()}")
    return r.stdout.strip()


# ─── Direct API call ──────────────────────────────────────────────────────────

def _call_api(messages, token, max_tokens=150, max_retries=6):
    project  = os.getenv("GOOGLE_CLOUD_PROJECT", "text2sql-research")
    region   = os.getenv("VERTEX_REGION", "us-east5")
    model    = os.getenv("VERTEX_MODEL", "meta/llama-4-maverick-17b-128e-instruct-maas")
    endpoint = (f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/"
                f"{project}/locations/{region}/endpoints/openapi/chat/completions")

    payload = json.dumps({
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": messages,
    }).encode()

    for attempt in range(max_retries):
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (2 ** attempt) * 4   # 4, 8, 16, 32, 64, 128 seconds
                print(f"  ⏳ 429 rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif e.code == 401:
                raise RuntimeError("401 Unauthorized — run: gcloud auth application-default login")
            else:
                raise
        except Exception:
            raise
    raise RuntimeError(f"All {max_retries} retries exhausted (persistent 429)")


def _clean_sql(raw):
    sql = raw.split(";")[0].split("\n")[0].strip()
    sql = re.sub(r"```[a-z]*", "", sql).strip("`").strip()
    sql = re.sub(r"\s+", " ", sql).strip()
    # Fix: model sometimes outputs "SELECT ..." as completion when prefill is "SELECT "
    # → results in "SELECT SELECT ..." — strip the duplicate
    sql = re.sub(r"^SELECT\s+SELECT\b", "SELECT", sql, flags=re.IGNORECASE)
    # Also fix triple/multiple SELECT
    while re.match(r"^SELECT\s+SELECT\b", sql, re.IGNORECASE):
        sql = re.sub(r"^SELECT\s+SELECT\b", "SELECT", sql, flags=re.IGNORECASE)
    return sql.strip()


# ─── Schema helpers ───────────────────────────────────────────────────────────

def get_schema_text(db_id, db_dir, questions_data, idx):
    """
    Return sanitized column names that match the SQLite database and gold SQL.
    CRITICAL: must use SQLite PRAGMA names (safe/sanitized), NOT original headers.
    Original headers like "# in Series" cause parse errors in generated SQL.
    Safe names like "no_in_series" match both the DB and the few-shot gold SQL.
    """
    # PRIMARY: SQLite PRAGMA — returns the exact safe column names used in SQL
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    if os.path.exists(db_path):
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            cur  = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            if tables:
                cur.execute(f"PRAGMA table_info({tables[0]})")
                cols = [r[1] for r in cur.fetchall()]
                conn.close()
                if cols:
                    return ", ".join(cols)
            conn.close()
        except Exception:
            pass
    # FALLBACK: original headers (only if SQLite unavailable)
    if idx < len(questions_data):
        tbl = questions_data[idx].get("table", {})
        if isinstance(tbl, dict):
            hdrs = tbl.get("header", [])
            if hdrs:
                return ", ".join(hdrs)
    return ""


# ─── SQL validation ──────────────────────────────────────────────────────────

def _sql_executes(sql, db_id, db_dir):
    """Try to execute SQL against the SQLite database. Return True if no error."""
    import sqlite3
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return True  # can't validate, accept anyway
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(sql)
        conn.close()
        return True
    except Exception:
        try: conn.close()
        except Exception: pass
        return False


# ─── Mode 2: Few-shot same-table regeneration ─────────────────────────────────

def build_gold_index(gold_file):
    """Build {question → sql} and {db_id → [items]} maps from gold file."""
    with open(gold_file, encoding="utf-8") as f:
        data = json.load(f)
    q2sql   = {}
    db2items = {}
    for item in data:
        q   = item.get("question", "")
        sql = item.get("query") or item.get("sql", "")
        # If sql is a dict (original dev.json format), skip — need string form
        if isinstance(sql, str) and sql.strip():
            q2sql[q] = sql
        db_id = item.get("db_id", "")
        if db_id:
            db2items.setdefault(db_id, []).append(item)
    return q2sql, db2items


def regen_one_few_shot(line_no, questions_data, db_dir, token,
                       q2sql, db2items, n_shots=4):
    """
    Regenerate with same-table few-shot examples.
    Shows the model 4 gold Q→SQL pairs from the same table so it learns:
    - exact value format used in this table
    - whether COUNT/MAX/bare is used for 'how many' queries
    - correct column names
    """
    idx = line_no - 1
    if idx < 0 or idx >= len(questions_data):
        return {"line": line_no, "ok": False, "sql": "SELECT 1",
                "tsv_row": "SELECT 1\t", "error": "out of range",
                "db_id": "", "question": ""}

    item     = questions_data[idx]
    question = item.get("question", "")
    db_id    = item.get("db_id", "")
    schema   = get_schema_text(db_id, db_dir, questions_data, idx)

    # Gather same-table gold examples (exclude the current question)
    siblings = [
        it for it in db2items.get(db_id, [])
        if it.get("question") != question
           and isinstance(it.get("query") or it.get("sql", ""), str)
    ]
    # Prefer examples that show diverse AGG (COUNT, MAX, bare)
    examples = siblings[:n_shots]

    # Build few-shot block
    shot_block = ""
    for ex in examples:
        ex_q   = ex.get("question", "")
        ex_sql = ex.get("query") or ex.get("sql", "")
        if ex_q and ex_sql and isinstance(ex_sql, str):
            shot_block += f"Q: {ex_q}\nSQL: {ex_sql}\n\n"

    user_prompt = (
        f"You are a WikiSQL SQL expert. Generate correct SQL for the question below.\n\n"
        f"Table: wikisql_data\n"
        f"Columns: {schema}\n\n"
    )
    if shot_block:
        user_prompt += f"Examples from the SAME table:\n{shot_block}"

    user_prompt += (
        f"Rules:\n"
        f"- Table name is always: wikisql_data\n"
        f"- Use exact column names from the schema\n"
        f"- String values must match the format shown in examples\n"
        f"- No semicolons, no markdown\n\n"
        f"Question: {question}\n"
        f"SQL:"
    )

    messages = [
        {"role": "system",
         "content": "Output ONLY a single SQL SELECT statement. No explanations."},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": "SELECT "},
    ]

    try:
        completion = _call_api(messages, token)
        full_sql   = _clean_sql("SELECT " + completion)

        if (full_sql.upper().startswith("SELECT")
                and "wikisql_data" in full_sql.lower()
                and _sql_executes(full_sql, db_id, db_dir)):
            return {"line": line_no, "ok": True, "sql": full_sql,
                    "tsv_row": f"{full_sql}\t{db_id}",
                    "error": "", "db_id": db_id, "question": question}
        return {"line": line_no, "ok": False, "sql": "",
                "tsv_row": "",
                "error": f"Bad/unexecutable output: {full_sql!r}",
                "db_id": db_id, "question": question}
    except Exception as e:
        return {"line": line_no, "ok": False, "sql": "",
                "tsv_row": "",
                "error": str(e), "db_id": db_id, "question": question}


def load_lines_from_failures_csv(csv_path, fail_cats=None):
    """
    Read line numbers from em_failures.csv.
    fail_cats: list of fail_cat values to include. None = all.
    Default targets: agg, cond, sel, agg+sel, agg+cond, sel+cond, agg+sel+cond
    """
    if fail_cats is None:
        fail_cats = {"agg", "cond", "sel",
                     "agg+sel", "agg+cond", "sel+cond", "agg+sel+cond"}
    lines = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("fail_cat", "")
            if cat in fail_cats:
                try:
                    lines.append(int(row["line_no"]))
                except (ValueError, KeyError):
                    pass
    return sorted(set(lines))


# ─── Mode 3: Force COUNT ──────────────────────────────────────────────────────

_COUNT_TRIGGER = re.compile(
    r"\b(how many|number of|total number of|name the number|how often|in how many)\b",
    re.IGNORECASE)

_NUMERIC_QTY = re.compile(
    r"\b(goal|goals|viewer|viewers|vote|votes|point|points|run|runs|"
    r"lap|laps|yard|yards|score|scores|seat|seats|passenger|passengers|"
    r"revenue|budget|crowd|rating|attendance|medal|strike|error|"
    r"assist|rebound|minute|win|wins|loss|losses|draw|draws|tie|ties|"
    r"penalty|foul|card|cap|caps|pass|passes|shot|shots|"
    r"wicket|over|overs|inning|innings|start|appearance)\b",
    re.IGNORECASE)

_BARE_SELECT = re.compile(
    r"^SELECT\s+(\w+)\s+FROM\s+(\w+)((?:\s+WHERE\s+.+?)?)\s*$",
    re.IGNORECASE | re.DOTALL)

_HAS_AGG = re.compile(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", re.IGNORECASE)


def is_count_target(question, sql):
    if _HAS_AGG.search(sql): return False
    if not _COUNT_TRIGGER.search(question): return False
    m = _BARE_SELECT.match(sql.strip())
    if not m: return False
    return not _NUMERIC_QTY.search(m.group(1))


def find_count_target_lines(rows, questions):
    return [i+1 for i, row in enumerate(rows)
            if is_count_target(
                questions[i] if i < len(questions) else "",
                row.split("\t")[0].strip())]


def regen_one_count_forced(line_no, questions_data, db_dir, token):
    idx = line_no - 1
    if idx < 0 or idx >= len(questions_data):
        return {"line": line_no, "ok": False, "sql": "SELECT 1",
                "tsv_row": "SELECT 1\t", "error": "out of range",
                "db_id": "", "question": ""}

    item     = questions_data[idx]
    question = item.get("question", "")
    db_id    = item.get("db_id", "")
    schema   = get_schema_text(db_id, db_dir, questions_data, idx)

    user_prompt = (
        f"Complete this WikiSQL COUNT query.\n\n"
        f"Table columns: {schema}\n"
        f"Question: {question}\n\n"
        f"Rules:\n"
        f"- Table name: wikisql_data\n"
        f"- Use a column from the list above\n"
        f"- Add WHERE conditions from the question\n"
        f"- Format: COUNT(col) FROM wikisql_data [WHERE col = 'value']\n"
        f"- No semicolons. No markdown.\n\n"
        f"Complete after SELECT COUNT:"
    )

    messages = [
        {"role": "system",
         "content": "You complete SQL queries. Output only the SQL continuation."},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": "SELECT COUNT("},
    ]

    try:
        completion = _call_api(messages, token, max_tokens=120)
        full_sql   = _clean_sql("SELECT COUNT(" + completion)
        if "COUNT(" in full_sql.upper() and "wikisql_data" in full_sql.lower():
            return {"line": line_no, "ok": True, "sql": full_sql,
                    "tsv_row": f"{full_sql}\t{db_id}",
                    "error": "", "db_id": db_id, "question": question}
        return {"line": line_no, "ok": False, "sql": "SELECT 1",
                "tsv_row": f"SELECT 1\t{db_id}",
                "error": f"Bad output: {full_sql!r}",
                "db_id": db_id, "question": question}
    except Exception as e:
        return {"line": line_no, "ok": False, "sql": "SELECT 1",
                "tsv_row": f"SELECT 1\t{db_id}",
                "error": str(e), "db_id": db_id, "question": question}


# ─── Mode 1: Standard (SELECT 1 / pipeline) ───────────────────────────────────

def init_pipelines(args):
    from utils.sql_schema import load_full_db_context  # noqa
    semantic_pipeline = reasoning_pipeline = None

    if args.use_semantic:
        try:
            from src.semantic.semantic_pipeline import SemanticPipeline
            cfg = load_config(args.semantic_config) if args.semantic_config else {"enabled": True}
            semantic_pipeline = SemanticPipeline(cfg)
            print("✓ Semantic pipeline ready")
        except Exception as e:
            print(f"⚠ Semantic pipeline failed: {e}")

    if args.use_reasoning_bank:
        try:
            from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
            cfg = load_config(args.reasoning_config) or {}
            pc  = cfg.get("pipeline", {})
            reasoning_pipeline = ReasoningBankPipeline(
                db_path=pc.get("db_path", "./memory/reasoning_bank.db"),
                chromadb_path=pc.get("chromadb_path", "./memory/chromadb"),
                config=cfg)
            print("✓ ReasoningBank ready")
        except Exception as e:
            print(f"⚠ ReasoningBank failed: {e}")

    try:
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        print("✓ SQLGenerator ready")
    except Exception as e:
        print(f"✗ SQLGenerator failed: {e}")
        sys.exit(1)

    return semantic_pipeline, reasoning_pipeline, sql_generator


def regen_one(line_no, questions_data, db_dir,
              semantic_pipeline, reasoning_pipeline, sql_generator,
              max_retries=3):
    from utils.sql_schema import load_full_db_context
    idx = line_no - 1
    if idx < 0 or idx >= len(questions_data):
        return {"line": line_no, "db_id": "", "question": "",
                "sql": "SELECT 1", "tsv_row": "SELECT 1\t",
                "ok": False, "error": f"Line {line_no} out of range"}

    item     = questions_data[idx]
    question = item.get("question", "")
    db_id    = item.get("db_id", "unknown")
    db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        return {"line": line_no, "db_id": db_id, "question": question,
                "sql": "SELECT 1", "tsv_row": f"SELECT 1\t{db_id}",
                "ok": False, "error": f"DB not found: {db_path}"}

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            enhanced = question
            if semantic_pipeline:
                try:
                    res = semantic_pipeline.enhance_question(question, db_id, None)
                    enhanced = res.get("enhanced_question", question)
                except Exception:
                    pass

            sql = ""
            if reasoning_pipeline:
                try:
                    ctx = load_full_db_context(db_id, db_dir)
                    rb  = reasoning_pipeline.generate_with_reasoning(
                        question=enhanced, db_id=db_id,
                        schema=ctx.get("schema", {}),
                        gold_sql=item.get("query", item.get("sql")),
                        sql_generator=lambda q: sql_generator.generate(q, db_path))
                    sql = rb.get("sql", "") or ""
                except Exception as e:
                    last_error = str(e)

            if not sql or sql.strip().upper() == "SELECT 1":
                sql = sql_generator.generate(enhanced, db_path)

            sql = (sql or "SELECT 1").replace("\n", " ").strip()
            if sql.upper() != "SELECT 1":
                return {"line": line_no, "db_id": db_id, "question": question,
                        "sql": sql, "tsv_row": f"{sql}\t{db_id}",
                        "ok": True, "error": ""}

            last_error = "Model returned SELECT 1"
            time.sleep(2 * attempt)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return {"line": line_no, "db_id": db_id, "question": question,
            "sql": "SELECT 1", "tsv_row": f"SELECT 1\t{db_id}",
            "ok": False, "error": f"All {max_retries} retries failed. {last_error}"}


# ─── Printer ──────────────────────────────────────────────────────────────────

def _print_result(r):
    icon = "✓" if r["ok"] else "✗"
    sep  = "─" * 60
    print(f"\n{sep}")
    print(f"  {icon} Line {r['line']}  |  db_id: {r['db_id']}")
    print(f"  Q : {r['question'][:90]}")
    if not r["ok"]:
        print(f"  ⚠ {r['error']}")
    print(f"  SQL: {r['sql'][:120]}")
    print(sep)


# ─── Batch runner (shared by all API modes) ───────────────────────────────────

def _run_batch(target_lines, rows, regen_fn, args, label=""):
    if not args.no_backup:
        bak = backup_tsv(args.predict)
        print(f"✓ Backup → {bak}")

    print("Getting auth token...")
    token = get_access_token()

    fixed = failed = skipped = 0
    failed_lines = []

    for rank, ln in enumerate(target_lines, 1):
        if ln > len(rows):
            print(f"⚠ Line {ln} beyond TSV length — skipping")
            continue

        # Resume support: skip lines that already have valid SQL
        current_sql = rows[ln - 1].split("\t")[0].strip()
        if args.skip_existing and current_sql.upper() not in ("SELECT 1", ""):
            skipped += 1
            continue

        if rank % 50 == 0:
            try: token = get_access_token()
            except Exception: pass

        r = regen_fn(ln, token)
        _print_result(r)

        if r["ok"]:
            rows[ln - 1] = r["tsv_row"]  # only update on success
            fixed += 1
            save_tsv(rows, args.predict)  # flush immediately
        else:
            failed += 1
            failed_lines.append(ln)
            # keep original SQL — do NOT overwrite with SELECT 1

        time.sleep(args.delay)

    # Save retry file for next run
    retry_file = args.predict + ".retry_lines.txt"
    if failed_lines:
        with open(retry_file, "w") as f:
            f.write("\n".join(map(str, failed_lines)) + "\n")
        print(f"\n  ⟳ Retry file saved → {retry_file}")
        print(f"    Next run: --from_retry_file {retry_file} --skip_existing")

    print(f"\n{'═'*60}")
    print(f"  {label}Fixed   : {fixed}")
    print(f"  {label}Failed  : {failed}  ← original SQL kept")
    print(f"  {label}Skipped : {skipped} (already fixed)")
    print(f"  Saved → {args.predict}")
    print(f"{'═'*60}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    # Paths
    parser.add_argument("--predict",   required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--db",        required=True)

    # Which lines
    parser.add_argument("--lines",             type=int, nargs="+", metavar="N")
    parser.add_argument("--from_failures_csv", default=None,
                        help="Read failing line numbers from em_failures.csv")
    parser.add_argument("--interactive",  action="store_true")
    parser.add_argument("--dry_run",      action="store_true")

    # Modes
    parser.add_argument("--few_shot_mode", action="store_true",
                        help="Regenerate with same-table few-shot gold examples (best for EX)")
    parser.add_argument("--gold_file",     default=None,
                        help="Path to dev_spider_format.json (required for --few_shot_mode)")
    parser.add_argument("--n_shots",       type=int, default=4,
                        help="Number of same-table examples for few-shot (default 4)")
    parser.add_argument("--fail_cats",     nargs="+",
                        default=["agg","cond","sel","agg+sel","agg+cond","sel+cond","agg+sel+cond"],
                        help="Failure categories to target (used with --from_failures_csv)")

    parser.add_argument("--force_count",  action="store_true",
                        help="Force COUNT for 'how many' bare-SELECT predictions")

    # Pipeline flags (for standard mode)
    parser.add_argument("--use_reasoning_bank", action="store_true")
    parser.add_argument("--use_chromadb",       action="store_true")
    parser.add_argument("--use_semantic",        action="store_true")
    parser.add_argument("--reasoning_config",    default="configs/reasoning_config.yaml")
    parser.add_argument("--semantic_config",     default=None)

    # Safety / resume
    parser.add_argument("--max_retries",     type=int,   default=3)
    parser.add_argument("--delay",           type=float, default=0.4)
    parser.add_argument("--no_backup",       action="store_true")
    parser.add_argument("--skip_existing",   action="store_true",
                        help="Skip lines that already have valid (non-SELECT-1) SQL — for resuming")
    parser.add_argument("--from_retry_file", default=None,
                        help="Load line numbers from a previous run's .retry_lines.txt file")

    args = parser.parse_args()

    if not Path(args.predict).exists():
        print(f"✗ File not found: {args.predict}"); sys.exit(1)

    rows = load_tsv(args.predict)
    print(f"✓ Loaded {len(rows)} rows from {args.predict}")

    with open(args.questions, encoding="utf-8") as f:
        questions_data = json.load(f)
    questions = [item.get("question", "") for item in questions_data]
    print(f"✓ Loaded {len(questions)} questions")

    # ── MODE: few_shot ────────────────────────────────────────────────────────
    if args.few_shot_mode:
        print("\n📚 Few-shot mode — same-table gold examples")

        gold_file = args.gold_file or args.questions
        print(f"  Gold file: {gold_file}")
        q2sql, db2items = build_gold_index(gold_file)
        print(f"  Gold index: {len(q2sql)} questions, {len(db2items)} tables")

        if args.from_failures_csv:
            target_lines = load_lines_from_failures_csv(
                args.from_failures_csv, set(args.fail_cats))
            print(f"  Lines from CSV ({args.from_failures_csv}): {len(target_lines)}")
        elif args.from_retry_file:
            with open(args.from_retry_file) as f:
                target_lines = sorted(int(l.strip()) for l in f if l.strip().isdigit())
            print(f"  Lines from retry file ({args.from_retry_file}): {len(target_lines)}")
        elif args.lines:
            target_lines = sorted(set(args.lines))
            print(f"  Lines from --lines: {len(target_lines)}")
        else:
            parser.error("--few_shot_mode requires --from_failures_csv, --from_retry_file, or --lines")

        if args.dry_run:
            for ln in target_lines[:20]:
                q = questions[ln-1] if ln-1 < len(questions) else "?"
                sql = rows[ln-1].split("\t")[0].strip()
                print(f"  Line {ln:>5}: {q[:60]}")
                print(f"           {sql[:70]}")
            print(f"  ... ({len(target_lines)} total)")
            return

        def regen_fn(ln, token):
            return regen_one_few_shot(
                ln, questions_data, args.db, token,
                q2sql, db2items, n_shots=args.n_shots)

        _run_batch(target_lines, rows, regen_fn, args, label="Few-shot ")
        return

    # ── MODE: force_count ─────────────────────────────────────────────────────
    if args.force_count:
        print("\n🔢 COUNT-forcing mode (assistant prefill 'SELECT COUNT(')")
        target_lines = args.lines or find_count_target_lines(rows, questions)
        print(f"  Targets: {len(target_lines)} lines")

        if args.dry_run:
            for ln in target_lines[:20]:
                sql = rows[ln-1].split("\t")[0].strip()
                print(f"  Line {ln:>5}: {questions[ln-1][:60]}")
                print(f"           {sql[:70]}")
            return

        def regen_fn(ln, token):
            return regen_one_count_forced(ln, questions_data, args.db, token)

        _run_batch(target_lines, rows, regen_fn, args, label="COUNT ")
        return

    # ── MODE: standard (SELECT 1 / pipeline) ──────────────────────────────────
    if args.lines:
        target_lines = sorted(set(args.lines))
        print(f"  Mode: specific lines → {len(target_lines)}")
    else:
        target_lines = find_failed_lines(rows)
        print(f"  Mode: auto SELECT 1 → {len(target_lines)} found")

    if not target_lines:
        print("✓ Nothing to re-generate."); return

    if args.dry_run:
        for ln in target_lines:
            row = rows[ln-1] if ln <= len(rows) else "<out of range>"
            print(f"  Line {ln:>5}: {row[:80]}")
        return

    if not args.no_backup:
        bak = backup_tsv(args.predict)
        print(f"✓ Backup → {bak}")

    print("\nInitialising pipelines…")
    sem, rb, sg = init_pipelines(args)
    print()

    if not args.interactive:
        fixed = still_failed = 0
        for ln in target_lines:
            if ln > len(rows):
                print(f"⚠ Line {ln} beyond TSV — skipping"); continue
            r = regen_one(ln, questions_data, args.db, sem, rb, sg,
                          max_retries=args.max_retries)
            _print_result(r)
            rows[ln-1] = r["tsv_row"]
            if r["ok"]: fixed += 1
            else: still_failed += 1
            save_tsv(rows, args.predict)
            time.sleep(args.delay)

        print(f"\n{'═'*60}")
        print(f"  Fixed        : {fixed}")
        print(f"  Still failed : {still_failed}")
        print(f"  Saved → {args.predict}")
        print(f"{'═'*60}")
        if still_failed:
            print(f"\n  Remaining: {find_failed_lines(rows)}")
    else:
        auto = find_failed_lines(rows)
        print(f"Interactive — {len(auto)} SELECT 1 lines. Line# / 'a' / 'q'")
        while True:
            try: raw = input("Line / 'a' / 'q': ").strip()
            except (EOFError, KeyboardInterrupt): print("\nBye."); break
            if raw.lower() in ("q","quit"): print("Bye."); break
            if raw.lower() == "a":
                targets = find_failed_lines(rows)
                if not targets: print("  ✓ No SELECT 1 left!"); continue
            elif raw.isdigit(): targets = [int(raw)]
            else: print(f"  ✗ Unknown: {raw!r}"); continue
            for ln in targets:
                if ln > len(rows): print("  ✗ Out of range"); continue
                r = regen_one(ln, questions_data, args.db, sem, rb, sg,
                              max_retries=args.max_retries)
                _print_result(r)
                rows[ln-1] = r["tsv_row"]
                save_tsv(rows, args.predict)
                time.sleep(args.delay)
            auto = find_failed_lines(rows)
            print(f"\n  Remaining SELECT 1: {len(auto)}")


if __name__ == "__main__":
    main()