#!/usr/bin/env python3
"""
force_count_regen.py
====================
Forces COUNT generation for "how many / number of" questions where the model
stubbornly returns bare SELECT.

Strategy: uses assistant pre-fill "SELECT COUNT(" so the model CANNOT deviate
from returning a COUNT query — it must continue with a column name.

Usage:
    python3 scripts/force_count_regen.py \
        --predict   results/predictions_wikisql_v2.tsv \
        --questions data/raw/wikisql/dev_spider_format.json \
        --db        data/raw/wikisql/database
"""

import os, sys, re, json, time, subprocess, argparse, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ── Patterns ──────────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_count_target(question: str, sql: str) -> bool:
    """True if this question should use COUNT but current pred is bare SELECT."""
    if _HAS_AGG.search(sql):
        return False                          # already has aggregation
    if not _COUNT_TRIGGER.search(question):
        return False                          # no "how many" trigger
    m = _BARE_SELECT.match(sql.strip())
    if not m:
        return False                          # not a bare SELECT
    col = m.group(1)
    if _NUMERIC_QTY.search(col):
        return False                          # numeric column → keep bare
    return True


def get_schema_text(db_id: str, db_dir: str, questions_data: list, idx: int) -> str:
    """Extract column names for the question's table."""
    item = questions_data[idx]
    # Try embedded table first (dev.json format)
    tbl = item.get("table", {})
    if tbl:
        headers = tbl.get("header", [])
        if headers:
            return ", ".join(headers)
    # Fall back to schema from SQLite
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    if os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if tables:
            cur.execute(f"PRAGMA table_info({tables[0]})")
            cols = [r[1] for r in cur.fetchall()]
            conn.close()
            return ", ".join(cols)
        conn.close()
    return ""


def get_access_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"gcloud auth failed: {result.stderr.strip()}")
    return result.stdout.strip()


def call_llm_count_forced(question: str, schema_text: str, token: str) -> str:
    """
    Call Vertex AI MaaS with 'SELECT COUNT(' as assistant prefill.
    The model MUST continue from that prefix → always returns a COUNT query.
    """
    import urllib.request

    project  = os.getenv("GOOGLE_CLOUD_PROJECT", "text2sql-research")
    region   = os.getenv("VERTEX_REGION", "us-east5")
    model    = "meta/llama-4-maverick-17b-128e-instruct-maas"
    endpoint = (f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/"
                f"{project}/locations/{region}/endpoints/openapi/chat/completions")

    system_prompt = (
        "You are a WikiSQL SQL generator. "
        "Output ONLY the remainder of a SQL query. "
        "No explanations. No markdown. No semicolons."
    )

    user_prompt = (
        f"Complete this SQL COUNT query for WikiSQL.\n\n"
        f"Table columns: {schema_text}\n"
        f"Question: {question}\n\n"
        f"Rules:\n"
        f"- Table name is always: wikisql_data\n"
        f"- Use one of the columns listed above\n"
        f"- Add WHERE conditions from the question\n"
        f"- Format: COUNT(column_name) FROM wikisql_data [WHERE ...]\n\n"
        f"Complete the query starting after SELECT COUNT:"
    )

    payload = json.dumps({
        "model":       model,
        "temperature": 0.0,
        "max_tokens":  120,
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": "SELECT COUNT("},  # force prefix
        ],
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            completion = data["choices"][0]["message"]["content"].strip()
            # Reconstruct full SQL
            full_sql = "SELECT COUNT(" + completion
            # Clean up: stop at first semicolon or newline
            full_sql = full_sql.split(";")[0].split("\n")[0].strip()
            return full_sql
    except Exception as e:
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Force COUNT generation for 'how many' questions")
    parser.add_argument("--predict",   required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--db",        required=True)
    parser.add_argument("--dry_run",   action="store_true", help="Show targets without calling LLM")
    parser.add_argument("--delay",     type=float, default=0.5)
    args = parser.parse_args()

    # Load predictions
    with open(args.predict, encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f]

    preds = [r.split("\t")[0].strip() for r in rows]
    db_ids = [r.split("\t")[1].strip() if "\t" in r else "" for r in rows]

    # Load questions
    with open(args.questions, encoding="utf-8") as f:
        questions_data = json.load(f)
    questions = [item.get("question", "") for item in questions_data]

    # Identify targets
    targets = []
    for i, (q, sql) in enumerate(zip(questions, preds)):
        if is_count_target(q, sql):
            targets.append(i)

    print(f"Found {len(targets)} COUNT targets")

    if args.dry_run:
        for i in targets[:20]:
            print(f"  Line {i+1}: {questions[i][:70]}")
            print(f"    pred: {preds[i][:80]}")
        return

    # Backup
    bak = args.predict + ".count_bak"
    shutil.copy2(args.predict, bak)
    print(f"✓ Backup: {bak}")

    # Get auth token
    print("Getting auth token...")
    token = get_access_token()

    fixed = failed = 0
    for rank, i in enumerate(targets, 1):
        q      = questions[i]
        sql    = preds[i]
        db_id  = db_ids[i]
        schema = get_schema_text(db_id, args.db, questions_data, i)

        print(f"\n[{rank}/{len(targets)}] Line {i+1}")
        print(f"  Q:    {q[:75]}")
        print(f"  old:  {sql[:80]}")

        new_sql = call_llm_count_forced(q, schema, token)

        if new_sql and "COUNT(" in new_sql.upper() and "wikisql_data" in new_sql.lower():
            new_sql = re.sub(r"\s+", " ", new_sql).strip().rstrip(";")
            print(f"  new:  {new_sql[:80]}")
            rows[i] = f"{new_sql}\t{db_id}" if db_id else new_sql
            preds[i] = new_sql
            fixed += 1
        else:
            print(f"  ✗ LLM returned unusable result: {new_sql!r}")
            failed += 1

        # Refresh token every 50 requests
        if rank % 50 == 0:
            try:
                token = get_access_token()
            except Exception:
                pass

        # Save after every fix
        with open(args.predict, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(row + "\n")

        time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"  Fixed  : {fixed}")
    print(f"  Failed : {failed}")
    print(f"  Saved  → {args.predict}")
    print(f"{'='*60}")
    print("\nRe-evaluate:")
    print("  python3 scripts/evaluate_wikisql.py \\")
    print("      --gold  data/raw/wikisql/dev_spider_format.json \\")
    print("      --table data/raw/wikisql/tables.json \\")
    print(f"      --predict {args.predict} \\")
    print("      --etype all")


if __name__ == "__main__":
    main()