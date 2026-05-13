#!/usr/bin/env python3
"""
Interactive DIN-SQL regenerator — input a line number, get the SQL back.
=========================================================================
Given one or more 1-based line numbers from the TSV, this script looks up
the matching question, runs the DIN-SQL pipeline, and prints the result so
you can manually paste it back into the TSV.

Usage (single line):
    python scripts/baselines/regen_line.py --line 42

Usage (multiple lines at once):
    python scripts/baselines/regen_line.py --line 42 57 103

Usage (interactive loop — keeps asking until you type 'q'):
    python scripts/baselines/regen_line.py --interactive

All paths default to the standard project layout but can be overridden.
"""

import sys
import os
import json
import argparse
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Import run_dinsql ─────────────────────────────────────────────────────────
import importlib.util, types

def _import_run_dinsql() -> types.ModuleType:
    candidates = [
        Path(__file__).parent / "run_dinsql.py",
        Path(__file__).parent.parent.parent / "scripts" / "baselines" / "run_dinsql.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("run_dinsql", p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("Cannot find run_dinsql.py. Place this script in scripts/baselines/.")

_dinsql      = _import_run_dinsql()
run_dinsql   = _dinsql.run_dinsql
_build_model = _dinsql._build_model


# ── Core logic ────────────────────────────────────────────────────────────────

def regen_line(
    line_no: int,           # 1-based line number (matches TSV row)
    questions: list,
    db_dir: str,
    model,
    wikisql_fast: bool = True,
    skip_selfcorrect: bool = False,
) -> dict:
    """
    Regenerate SQL for a single TSV line number.

    Returns a dict:
        {
            "line"     : 1-based line number,
            "db_id"    : "1-15418319-1",
            "question" : "What is the ...",
            "sql"      : "SELECT ... FROM wikisql_data WHERE ...",
            "tsv_row"  : "SELECT ...\t1-15418319-1",   ← ready to paste
            "error"    : ""   (non-empty if something went wrong)
        }
    """
    idx = line_no - 1   # convert to 0-based

    if idx < 0 or idx >= len(questions):
        return {
            "line": line_no, "db_id": "", "question": "",
            "sql": "", "tsv_row": "",
            "error": f"Line {line_no} is out of range (total questions: {len(questions)})",
        }

    q        = questions[idx]
    question = q.get("question", "")
    db_id    = q.get("db_id", "unknown")
    db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        return {
            "line": line_no, "db_id": db_id, "question": question,
            "sql": "SELECT 1", "tsv_row": f"SELECT 1\t{db_id}",
            "error": f"Database not found: {db_path}",
        }

    try:
        result = run_dinsql(
            model, db_path, question,
            skip_selfcorrect=skip_selfcorrect,
            wikisql_fast=wikisql_fast,
        )
        sql     = result["sql"].replace("\n", " ").strip()
        err_msg = result.get("error", "")
    except Exception as e:
        sql     = "SELECT 1"
        err_msg = str(e)

    tsv_row = f"{sql}\t{db_id}"

    return {
        "line"    : line_no,
        "db_id"   : db_id,
        "question": question,
        "sql"     : sql,
        "tsv_row" : tsv_row,
        "error"   : err_msg,
    }


def _save_results(results: list, path: str):
    """
    Write a clean file with one entry per result:

        === Line 611 | db_id: 1-15418319-1 ===
        Question : What is ...
        SQL      : SELECT ... FROM wikisql_data WHERE ...
        TSV ROW  : SELECT ...\t1-15418319-1
        [error]  : (only shown if non-empty)

    The TSV ROW line is what you paste directly into the predictions file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Regenerated {len(results)} line(s)\n")
        f.write("# Copy the TSV ROW line for each entry into your predictions file.\n\n")
        for r in results:
            f.write(f"=== Line {r['line']} | db_id: {r['db_id']} ===\n")
            f.write(f"Question : {r['question']}\n")
            f.write(f"SQL      : {r['sql']}\n")
            if r["error"]:
                f.write(f"[error]  : {r['error']}\n")
            f.write(f"TSV ROW  : {r['tsv_row']}\n")
            f.write("\n")


def _print_result(r: dict):
    """Pretty-print one result."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Line     : {r['line']}")
    print(f"  db_id    : {r['db_id']}")
    print(f"  Question : {r['question']}")
    if r["error"]:
        print(f"  ⚠ Error  : {r['error']}")
    print(f"\n  SQL      : {r['sql']}")
    print(f"\n  ✂ Paste this into the TSV (line {r['line']}):")
    print(f"  {r['tsv_row']}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate a single DIN-SQL prediction by TSV line number",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--line", type=int, nargs="+", metavar="N",
        help="1-based line number(s) to regenerate (e.g. --line 42 57 103)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive mode: keep asking for line numbers until you type 'q'",
    )
    parser.add_argument(
        "--questions",
        default="data/raw/wikisql/dev_spider_format.json",
        help="Path to spider-format questions JSON",
    )
    parser.add_argument(
        "--db",
        default="data/raw/wikisql/database",
        help="Database directory",
    )
    parser.add_argument("--wikisql_fast",     action="store_true", default=True)
    parser.add_argument("--no_wikisql_fast",  action="store_true",
                        help="Disable wikisql_fast (run full 4-stage pipeline)")
    parser.add_argument("--skip_selfcorrect", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="Save results to a file (e.g. --save results/regen_611_658.txt)")

    args = parser.parse_args()

    if not args.line and not args.interactive:
        parser.error("Provide --line <N> [N ...] or --interactive")

    wikisql_fast = args.wikisql_fast and not args.no_wikisql_fast

    # Set delay on the imported module
    _dinsql._INTER_CALL_DELAY = args.delay

    # Load questions
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"✓ Loaded {len(questions)} questions from {args.questions}")

    # Build model (once, shared across all regen calls)
    print("✓ Building model…")
    model = _build_model()
    print("✓ Model ready\n")

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.line:
        results = []
        for line_no in args.line:
            r = regen_line(
                line_no, questions, args.db, model,
                wikisql_fast=wikisql_fast,
                skip_selfcorrect=args.skip_selfcorrect,
            )
            _print_result(r)
            results.append(r)

        if args.save:
            _save_results(results, args.save)
            print(f"\n✓ Saved → {args.save}")

    # ── Interactive mode ──────────────────────────────────────────────────────
    if args.interactive:
        print("Interactive mode — enter a line number to regenerate, or 'q' to quit.")
        print(f"  Valid range: 1 – {len(questions)}\n")
        while True:
            try:
                raw = input("Line number (or 'q'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if raw.lower() in ("q", "quit", "exit", ""):
                print("Bye.")
                break

            if not raw.isdigit():
                print(f"  ✗ '{raw}' is not a valid number. Try again.")
                continue

            line_no = int(raw)
            r = regen_line(
                line_no, questions, args.db, model,
                wikisql_fast=wikisql_fast,
                skip_selfcorrect=args.skip_selfcorrect,
            )
            _print_result(r)


if __name__ == "__main__":
    main()