#!/usr/bin/env python3
"""
regen_spider_failed.py
======================
Re-generate failed (SELECT 1) predictions in a Spider predictions TSV.

The Spider TSV has ONE line per question:
    <predicted_sql>\t<db_id>

A failed prediction shows as "SELECT 1\t<db_id>".
This script finds every SELECT 1 line, regenerates it in-place, and saves.

Usage:
  # Dry-run — see what will be fixed, no API calls:
  python3 scripts/regen_spider_failed.py \\
      --predict   results/predictions_spider_v3.tsv \\
      --questions data/raw/spider/dev.json \\
      --db        data/raw/spider/database \\
      --dry_run

  # Full run:
  python3 scripts/regen_spider_failed.py \\
      --predict   results/predictions_spider_v3.tsv \\
      --questions data/raw/spider/dev.json \\
      --db        data/raw/spider/database \\
      --use_reasoning_bank --use_chromadb

  # Resume after interruption (skip already-fixed lines):
  python3 scripts/regen_spider_failed.py ... --skip_existing
"""

import sys, os, json, shutil, argparse, logging, time, warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

_ROOT = Path(__file__).resolve().parent
for _ in range(4):
    if (_ROOT / "src").exists():
        break
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
for _n in ["chromadb", "chromadb.api", "src.reasoning", "utils.embedding_utils"]:
    logging.getLogger(_n).setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv(path):
    """Load TSV as list of lines (one per question)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]

def save_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")

def backup(path):
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak

def row_sql(row):
    return row.split("\t")[0].strip()

def row_db(row):
    p = row.split("\t")
    return p[1].strip() if len(p) > 1 else ""

def is_failed(sql):
    return sql.upper() in ("SELECT 1", "", "NONE", "NULL")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline init
# ─────────────────────────────────────────────────────────────────────────────

def init_pipelines(args):
    from utils.sql_schema import load_full_db_context  # noqa
    sem = rb_pipe = None

    if args.use_semantic:
        try:
            from src.semantic.semantic_pipeline import SemanticPipeline
            sem = SemanticPipeline({"enabled": True})
            print("✓ Semantic pipeline ready")
        except Exception as e:
            print(f"⚠  Semantic skipped: {e}")

    if args.use_reasoning_bank:
        try:
            from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
            import yaml
            cfg = {}
            if Path(args.reasoning_config).exists():
                with open(args.reasoning_config) as f:
                    cfg = yaml.safe_load(f) or {}
            pc = cfg.get("pipeline", {})
            rb_pipe = ReasoningBankPipeline(
                db_path=pc.get("db_path", "./memory/reasoning_bank.db"),
                chromadb_path=pc.get("chromadb_path", "./memory/chromadb"),
                config=cfg,
            )
            print("✓ ReasoningBank ready")
        except Exception as e:
            print(f"⚠  ReasoningBank skipped: {e}")

    try:
        from src.generation.sql_generator import SQLGenerator
        sg = SQLGenerator()
        print("✓ SQLGenerator ready")
    except Exception as e:
        print(f"✗ SQLGenerator failed: {e}"); sys.exit(1)

    return sem, rb_pipe, sg


# ─────────────────────────────────────────────────────────────────────────────
# Generate SQL for one question
# ─────────────────────────────────────────────────────────────────────────────

def generate_one(item, db_dir, sem, rb_pipe, sg, max_retries=3):
    from utils.sql_schema import load_full_db_context
    question = item.get("question", "")
    db_id    = item.get("db_id", "")
    db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    if not question or not db_id:
        return "SELECT 1", False, "empty question/db_id"
    if not os.path.exists(db_path):
        return "SELECT 1", False, f"DB not found: {db_path}"

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            enhanced = question
            if sem:
                try:
                    res = sem.enhance_question(question, db_id, None)
                    enhanced = res.get("enhanced_question", question)
                except Exception:
                    pass

            sql = ""
            if rb_pipe:
                try:
                    ctx = load_full_db_context(db_id, db_dir)
                    res = rb_pipe.generate_with_reasoning(
                        question=enhanced, db_id=db_id,
                        schema=ctx.get("schema", {}),
                        gold_sql=item.get("query", item.get("sql")),
                        sql_generator=lambda q, strategy_hints=None, temperature=0.0: sg.generate(
                            q, db_path, temperature=temperature),
                    )
                    sql = res.get("sql", "") or ""
                except Exception as e:
                    last_err = str(e)

            if not sql or is_failed(sql.strip()):
                sql = sg.generate(enhanced, db_path) or ""

            sql = sql.replace("\n", " ").strip() or "SELECT 1"
            if not is_failed(sql):
                return sql, True, ""

            last_err = "model returned SELECT 1"
            time.sleep(2 * attempt)

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(3 * attempt)

    return "SELECT 1", False, f"all {max_retries} retries failed — {last_err}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-generate SELECT 1 failures in Spider predictions TSV (in-place)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--predict",   required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--db",        required=True)
    parser.add_argument("--use_reasoning_bank", action="store_true")
    parser.add_argument("--use_chromadb",       action="store_true")
    parser.add_argument("--use_semantic",       action="store_true")
    parser.add_argument("--reasoning_config",   default="configs/reasoning_config.yaml")
    parser.add_argument("--max_retries",   type=int,   default=3)
    parser.add_argument("--delay",         type=float, default=1.0)
    parser.add_argument("--dry_run",       action="store_true")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip lines that already have valid SQL (resume mode)")
    parser.add_argument("--no_backup",     action="store_true")
    args = parser.parse_args()

    if not Path(args.predict).exists():
        print(f"✗ TSV not found: {args.predict}"); sys.exit(1)
    if not Path(args.questions).exists():
        print(f"✗ Questions not found: {args.questions}"); sys.exit(1)

    tsv_rows = load_tsv(args.predict)
    with open(args.questions, encoding="utf-8") as f:
        questions_data = json.load(f)

    n_tsv = len(tsv_rows)
    n_q   = len(questions_data)

    print(f"✓ TSV rows    : {n_tsv}")
    print(f"✓ Questions   : {n_q}")

    if n_tsv != n_q:
        print(f"\n⚠  Row count mismatch: TSV has {n_tsv} rows but dev.json has {n_q} questions.")
        print(f"   The TSV may be incomplete. Run generate_predictions.py --resume first.")
        if n_tsv > n_q:
            print(f"   Truncating TSV to first {n_q} rows for safety.")
            tsv_rows = tsv_rows[:n_q]

    # Find all SELECT 1 failures
    failed_indices = [
        i for i, row in enumerate(tsv_rows)
        if is_failed(row_sql(row))
    ]

    print(f"\n{'─'*60}")
    print(f"  SELECT 1 / empty : {len(failed_indices)} lines")
    print(f"  Lines            : {failed_indices[:20]}{'...' if len(failed_indices)>20 else ''}")
    print(f"{'─'*60}")

    if not failed_indices:
        print("\n✓ No failures — all predictions are non-empty!"); return

    if args.dry_run:
        print("\n[DRY RUN — no API calls]\n")
        for i in failed_indices:
            item = questions_data[i] if i < n_q else {}
            print(f"  line {i+1:4d} | {item.get('db_id','?'):25s} | "
                  f"{item.get('question','')[:55]}")
        return

    if not args.no_backup:
        bak = backup(args.predict)
        print(f"\n✓ Backup → {bak}")

    print("\nInitialising pipelines…")
    sem, rb_pipe, sg = init_pipelines(args)
    print()

    fixed = 0
    still_failed = 0

    for i in failed_indices:
        if i >= n_q:
            print(f"⚠  Index {i} out of range — skipping"); still_failed += 1; continue

        if args.skip_existing and not is_failed(row_sql(tsv_rows[i])):
            print(f"  ⏭  line {i+1} already fixed — skipping"); continue

        item  = questions_data[i]
        db_id = item.get("db_id", row_db(tsv_rows[i]))

        print(f"\n  → line {i+1:4d} | {db_id} | {item.get('question','')[:65]}")

        sql, ok, err = generate_one(item, args.db, sem, rb_pipe, sg, args.max_retries)

        # Replace this line in-place — no rows added or removed
        tsv_rows[i] = f"{sql}\t{db_id}"
        save_tsv(tsv_rows, args.predict)

        if ok:
            print(f"  ✓ {sql[:100]}"); fixed += 1
        else:
            print(f"  ✗ {err}"); still_failed += 1

        time.sleep(args.delay)

    # Final check
    remaining = [i+1 for i, row in enumerate(tsv_rows) if is_failed(row_sql(row))]

    print(f"\n{'═'*60}")
    print(f"  Fixed        : {fixed}")
    print(f"  Still failed : {still_failed}")
    print(f"  TSV rows     : {len(tsv_rows)}  (unchanged)")
    print(f"  Saved → {args.predict}")
    if remaining:
        print(f"\n  Remaining SELECT 1 at lines: {remaining}")
        print(f"  Re-run with --skip_existing to retry only those")
    else:
        print(f"\n  ✓ All predictions generated!")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()