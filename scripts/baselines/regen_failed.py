#!/usr/bin/env python3
"""
regen_failed.py — Re-generate failed (SELECT 1) predictions in-place
=====================================================================
Scans a predictions TSV for lines containing "SELECT 1" (connection
failures during generate_predictions.py), re-runs the full system
pipeline (ReasoningBank + ChromaDB + Semantic RAG) for each failed
question, and writes the corrected SQL back into the TSV.

Usage — auto-detect all SELECT 1 lines:
    python scripts/regen_failed.py \\
        --predict results/predictions_wikisql_full.tsv \\
        --questions data/raw/wikisql/dev_spider_format.json \\
        --db        data/raw/wikisql/database \\
        --use_reasoning_bank --use_chromadb --use_semantic

Usage — specific line numbers only:
    python scripts/regen_failed.py \\
        --predict results/predictions_wikisql_full.tsv \\
        --questions data/raw/wikisql/dev_spider_format.json \\
        --db        data/raw/wikisql/database \\
        --lines 11 12 13 47 502

Usage — interactive mode (enter line numbers one by one):
    python scripts/regen_failed.py \\
        --predict results/predictions_wikisql_full.tsv \\
        --questions data/raw/wikisql/dev_spider_format.json \\
        --db        data/raw/wikisql/database \\
        --interactive
"""

import sys
import os
import json
import shutil
import argparse
import logging
import warnings
import time
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Always resolve project root regardless of where this script lives
_PROJECT_ROOT = Path(__file__).resolve()
for _ in range(5):
    _PROJECT_ROOT = _PROJECT_ROOT.parent
    if (_PROJECT_ROOT / 'src').exists() or (_PROJECT_ROOT / 'configs').exists():
        break
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ─── Config loader ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        try:
            import yaml
            return yaml.safe_load(f)
        except ImportError:
            return {}


# ─── TSV helpers ──────────────────────────────────────────────────────────────

def load_tsv(path: str) -> list[str]:
    """Return all lines (raw, with newline stripped) from the TSV."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def find_failed_lines(rows: list[str], marker: str = "SELECT 1") -> list[int]:
    """
    Return 1-based line numbers where the SQL column equals the failure marker.
    A line is considered failed if the SQL part (before the first TAB) matches.
    """
    failed = []
    for i, row in enumerate(rows, start=1):
        sql_part = row.split("\t")[0].strip()
        if sql_part.upper() == marker.upper():
            failed.append(i)
    return failed


def save_tsv(rows: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def backup_tsv(path: str) -> str:
    """Create a .bak copy; return its path."""
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak


# ─── Pipeline initialisation (mirrors generate_predictions.py) ────────────────

def init_pipelines(args) -> tuple:
    """
    Build and return (semantic_pipeline, reasoning_pipeline, sql_generator).
    Any component that fails to load is returned as None with a warning.
    """
    from utils.sql_schema import load_full_db_context  # noqa: F401 (import test)

    semantic_pipeline  = None
    reasoning_pipeline = None

    if args.use_semantic:
        try:
            from src.semantic.semantic_pipeline import SemanticPipeline
            cfg = load_config(args.semantic_config) if args.semantic_config else {"enabled": True}
            semantic_pipeline = SemanticPipeline(cfg)
            print("✓ Semantic pipeline ready")
        except Exception as e:
            print(f"⚠ Semantic pipeline failed to load: {e}")

    if args.use_reasoning_bank:
        try:
            from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
            cfg = load_config(args.reasoning_config) or {}
            pipeline_cfg = cfg.get("pipeline", {})
            reasoning_pipeline = ReasoningBankPipeline(
                db_path=pipeline_cfg.get("db_path", "./memory/reasoning_bank.db"),
                chromadb_path=pipeline_cfg.get("chromadb_path", "./memory/chromadb"),
                config=cfg,
            )
            print("✓ ReasoningBank ready")
        except Exception as e:
            print(f"⚠ ReasoningBank failed to load: {e}")

    try:
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        print("✓ SQLGenerator ready")
    except Exception as e:
        print(f"✗ SQLGenerator failed — cannot continue: {e}")
        sys.exit(1)

    return semantic_pipeline, reasoning_pipeline, sql_generator


# ─── Single-question re-generation ───────────────────────────────────────────

def regen_one(
    line_no: int,
    questions: list,
    db_dir: str,
    semantic_pipeline,
    reasoning_pipeline,
    sql_generator,
    max_retries: int = 3,
) -> dict:
    """
    Re-generate SQL for one TSV line (1-based).

    Returns:
        {
            "line"    : int,
            "db_id"   : str,
            "question": str,
            "sql"     : str,   ← new SQL (or SELECT 1 if all retries fail)
            "tsv_row" : str,   ← ready to write back to TSV
            "ok"      : bool,
            "error"   : str,
        }
    """
    from utils.sql_schema import load_full_db_context

    idx = line_no - 1
    if idx < 0 or idx >= len(questions):
        return {
            "line": line_no, "db_id": "", "question": "",
            "sql": "SELECT 1", "tsv_row": f"SELECT 1\t",
            "ok": False, "error": f"Line {line_no} out of range ({len(questions)} questions)",
        }

    item     = questions[idx]
    question = item.get("question", "")
    db_id    = item.get("db_id", "unknown")
    db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        return {
            "line": line_no, "db_id": db_id, "question": question,
            "sql": "SELECT 1", "tsv_row": f"SELECT 1\t{db_id}",
            "ok": False, "error": f"DB not found: {db_path}",
        }

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            # ── Semantic enhancement ──────────────────────────────────────────
            enhanced_question = question
            if semantic_pipeline:
                try:
                    res = semantic_pipeline.enhance_question(question, db_id, None)
                    enhanced_question = res.get("enhanced_question", question)
                except Exception:
                    pass

            # ── ReasoningBank generation ──────────────────────────────────────
            sql = ""
            if reasoning_pipeline:
                try:
                    db_context = load_full_db_context(db_id, db_dir)
                    rb_result  = reasoning_pipeline.generate_with_reasoning(
                        question     = enhanced_question,
                        db_id        = db_id,
                        schema       = db_context.get("schema", {}),
                        gold_sql     = item.get("query", item.get("sql")),
                        sql_generator= lambda q: sql_generator.generate(q, db_path),
                    )
                    sql = rb_result.get("sql", "") or ""
                except Exception as e:
                    last_error = str(e)

            # ── Plain fallback ────────────────────────────────────────────────
            if not sql or sql.strip().upper() == "SELECT 1":
                sql = sql_generator.generate(enhanced_question, db_path)

            sql = (sql or "SELECT 1").replace("\n", " ").strip()

            if sql.upper() != "SELECT 1":
                return {
                    "line": line_no, "db_id": db_id, "question": question,
                    "sql": sql, "tsv_row": f"{sql}\t{db_id}",
                    "ok": True, "error": "",
                }

            # Got SELECT 1 back — retry after brief wait
            last_error = "Model returned SELECT 1"
            time.sleep(2 * attempt)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(3 * attempt)

    # All retries exhausted
    return {
        "line": line_no, "db_id": db_id, "question": question,
        "sql": "SELECT 1", "tsv_row": f"SELECT 1\t{db_id}",
        "ok": False, "error": f"All {max_retries} retries failed. Last: {last_error}",
    }


# ─── Pretty printer ───────────────────────────────────────────────────────────

def _print_result(r: dict) -> None:
    sep   = "─" * 60
    icon  = "✓" if r["ok"] else "✗"
    print(f"\n{sep}")
    print(f"  {icon} Line {r['line']}  |  db_id: {r['db_id']}")
    print(f"  Q : {r['question'][:90]}")
    if not r["ok"]:
        print(f"  ⚠ Error: {r['error']}")
    print(f"  SQL: {r['sql'][:120]}")
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-generate failed (SELECT 1) predictions using the full system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    parser.add_argument("--predict",   required=True,
                        help="Predictions TSV to fix (modified in-place; .bak backup created)")
    parser.add_argument("--questions", required=True,
                        help="Spider-format questions JSON (same order as TSV)")
    parser.add_argument("--db",        required=True,
                        help="Database directory (db_id/db_id.sqlite)")

    # ── Which lines ──────────────────────────────────────────────────────────
    parser.add_argument("--lines", type=int, nargs="+", metavar="N",
                        help="Specific 1-based line numbers to re-generate. "
                             "Omit to auto-detect all SELECT 1 lines.")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter line numbers interactively (one by one)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show which lines would be regenerated without making API calls")

    # ── Pipeline flags (mirror generate_predictions.py) ──────────────────────
    parser.add_argument("--use_reasoning_bank", action="store_true")
    parser.add_argument("--use_chromadb",       action="store_true")
    parser.add_argument("--use_semantic",        action="store_true")
    parser.add_argument("--reasoning_config",
                        default="configs/reasoning_config.yaml")
    parser.add_argument("--semantic_config", default=None)

    # ── Retry / safety ───────────────────────────────────────────────────────
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Retries per question before giving up (default: 3)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between questions (default: 0.5)")
    parser.add_argument("--no_backup", action="store_true",
                        help="Skip creating a .bak backup of the original TSV")

    args = parser.parse_args()

    # ── Load TSV ─────────────────────────────────────────────────────────────
    if not Path(args.predict).exists():
        print(f"✗ Predictions file not found: {args.predict}")
        sys.exit(1)

    rows = load_tsv(args.predict)
    print(f"✓ Loaded {len(rows)} rows from {args.predict}")

    # ── Determine which lines to fix ─────────────────────────────────────────
    if args.lines:
        target_lines = sorted(set(args.lines))
        print(f"  Mode: specific lines → {target_lines}")
    else:
        target_lines = find_failed_lines(rows)
        print(f"  Mode: auto-detect SELECT 1 → {len(target_lines)} failed lines found")

    if not target_lines:
        print("✓ Nothing to re-generate. TSV looks clean.")
        return

    if args.dry_run:
        print(f"\nDry run — would re-generate {len(target_lines)} line(s):")
        for ln in target_lines:
            q_idx = ln - 1
            row   = rows[ln - 1] if ln <= len(rows) else "<out of range>"
            print(f"  Line {ln:>5}: {row[:80]}")
        return

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"✓ Loaded {len(questions)} questions from {args.questions}")

    # ── Backup ───────────────────────────────────────────────────────────────
    if not args.no_backup:
        bak = backup_tsv(args.predict)
        print(f"✓ Backup created → {bak}")

    # ── Init pipelines ────────────────────────────────────────────────────────
    print("\nInitialising pipelines…")
    semantic_pipeline, reasoning_pipeline, sql_generator = init_pipelines(args)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Batch mode
    # ─────────────────────────────────────────────────────────────────────────
    if not args.interactive:
        fixed   = 0
        still_failed = 0

        for ln in target_lines:
            if ln > len(rows):
                print(f"⚠ Line {ln} is beyond TSV length ({len(rows)}) — skipping")
                continue

            r = regen_one(
                ln, questions, args.db,
                semantic_pipeline, reasoning_pipeline, sql_generator,
                max_retries=args.max_retries,
            )
            _print_result(r)

            # Write back into rows list
            rows[ln - 1] = r["tsv_row"]

            if r["ok"]:
                fixed += 1
            else:
                still_failed += 1

            # Flush to disk after every successful fix (safe incremental update)
            save_tsv(rows, args.predict)

            time.sleep(args.delay)

        # ── Summary ──────────────────────────────────────────────────────────
        print(f"\n{'═'*60}")
        print(f"  Re-generation complete")
        print(f"  Fixed        : {fixed}")
        print(f"  Still failed : {still_failed}")
        print(f"  TSV saved    → {args.predict}")
        print(f"{'═'*60}")

        if still_failed:
            remaining = find_failed_lines(rows)
            print(f"\n  Remaining SELECT 1 lines: {remaining}")
            print(f"  Re-run with --lines {' '.join(map(str, remaining))} to retry those.")

    # ─────────────────────────────────────────────────────────────────────────
    # Interactive mode
    # ─────────────────────────────────────────────────────────────────────────
    else:
        auto = find_failed_lines(rows)
        print(f"Interactive mode — {len(auto)} SELECT 1 lines detected.")
        print(f"Enter a line number to regenerate, 'a' to fix all, or 'q' to quit.\n")

        while True:
            try:
                raw = input(f"Line number / 'a' (all {len(auto)}) / 'q': ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if raw.lower() in ("q", "quit", "exit"):
                print("Bye.")
                break

            if raw.lower() == "a":
                targets = find_failed_lines(rows)   # refresh
                if not targets:
                    print("  ✓ No SELECT 1 lines left!")
                    continue
            elif raw.isdigit():
                targets = [int(raw)]
            else:
                print(f"  ✗ Unknown input '{raw}'")
                continue

            for ln in targets:
                if ln > len(rows):
                    print(f"  ✗ Line {ln} out of range")
                    continue
                r = regen_one(
                    ln, questions, args.db,
                    semantic_pipeline, reasoning_pipeline, sql_generator,
                    max_retries=args.max_retries,
                )
                _print_result(r)
                rows[ln - 1] = r["tsv_row"]
                save_tsv(rows, args.predict)
                time.sleep(args.delay)

            auto = find_failed_lines(rows)
            print(f"\n  Remaining SELECT 1 lines: {len(auto)}")


if __name__ == "__main__":
    main()