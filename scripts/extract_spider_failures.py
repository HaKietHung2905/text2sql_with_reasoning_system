#!/usr/bin/env python3
"""
extract_spider_failures.py — Extract Spider evaluation failures to CSV
=======================================================================

Loads gold (dev.json) + predictions (.tsv), runs execution comparison,
and writes a failures CSV with columns:
  idx, db_id, question, hardness, fail_type, gold_sql, pred_sql, error_msg

fail_type values:
  exec_error    — predicted SQL raised a runtime exception
  exec_mismatch — SQL executed but results differ from gold
  em_only       — execution passed but exact match failed

Usage:
  python extract_spider_failures.py \\
      --gold    data/raw/spider/dev.json \\
      --db      data/raw/spider/database \\
      --predict results/predictions_spider.tsv \\
      --output  results/spider_failures.csv

Optional:
  --hardness hard extra   # filter to specific hardness levels
  --limit    200          # only process first N examples
"""

import sys
import os
import json
import sqlite3
import argparse
import csv
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Hardness detection (keyword-based, no Spider parser needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hardness(sql: str) -> str:
    """
    Classify SQL difficulty using Spider benchmark heuristics.
    Matches Spider's eval_hardness() closely without needing the full parser.
    """
    s = sql.lower()
    n_join     = len(re.findall(r'\bjoin\b', s))
    n_subquery = max(0, len(re.findall(r'\bselect\b', s)) - 1)
    has_group  = bool(re.search(r'\bgroup\s+by\b', s))
    has_having = bool(re.search(r'\bhaving\b', s))
    has_set_op = bool(re.search(r'\b(intersect|except|union)\b', s))
    n_cond     = len(re.findall(r'\b(and|or)\b', s))

    if has_set_op or n_subquery >= 2 or (n_join >= 2 and has_having):
        return 'extra'
    if n_subquery == 1 or n_join >= 2 or (has_group and n_join >= 1):
        return 'hard'
    if has_group or has_having or n_join == 1 or n_cond >= 2:
        return 'medium'
    return 'easy'


# ─────────────────────────────────────────────────────────────────────────────
# SQL execution helper
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(v):
    """Normalise a result cell for comparison: strip strings, round floats."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 4)
    return v


def execute_sql(sql: str, db_path: str):
    """
    Execute SQL against the given SQLite database.
    Returns (results, error_msg). results=None on error.
    Results are sorted and cell-normalised for order-insensitive comparison.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cur  = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        rows = sorted([tuple(_coerce(c) for c in row) for row in rows])
        return rows, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Exact-match comparison (normalised string-level)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(sql: str) -> str:
    """
    Normalise SQL for string-level EM comparison.

    Eliminates false failures caused by formatting differences between
    gold SQL (which uses extra spaces around commas, e.g. "name ,  age")
    and predicted SQL (which uses standard spacing, e.g. "name, age").
    Both are semantically identical and should not count as EM failures.

    Normalizations:
      1. Lowercase
      2. Strip trailing semicolons
      3. Collapse all whitespace to single space
      4. Normalise comma spacing:   " , "  /  "  ,"  →  ", "
      5. Normalise operator spacing: " = " / "  =  " → " = "
      6. Normalise paren spacing:   " ( " / "( " → "("
    """
    sql = sql.lower().strip().rstrip(';').strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*(!=|<=|>=|=|<|>)\s*', r' \1 ', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql


def em_match(pred: str, gold: str) -> bool:
    """Return True if pred and gold are EM-equivalent after normalisation."""
    return _norm(pred) == _norm(gold)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract Spider evaluation failures to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--gold',     required=True, help='Path to dev.json')
    ap.add_argument('--db',       required=True, help='Path to database directory')
    ap.add_argument('--predict',  required=True, help='Path to predictions .tsv')
    ap.add_argument('--output',   default='results/spider_failures.csv')
    ap.add_argument('--hardness', nargs='*',
                    choices=['easy', 'medium', 'hard', 'extra'],
                    default=None,
                    help='Filter to these hardness levels (default: all)')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process first N examples')
    args = ap.parse_args()

    # ── Load gold ─────────────────────────────────────────────────────────────
    with open(args.gold) as f:
        gold_entries = json.load(f)
    if args.limit:
        gold_entries = gold_entries[:args.limit]
    print(f"Loaded {len(gold_entries)} gold entries")

    # ── Load predictions ──────────────────────────────────────────────────────
    predictions = []
    with open(args.predict) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(line.split('\t')[0].strip())

    if len(predictions) < len(gold_entries):
        print(f"WARNING: {len(predictions)} predictions vs {len(gold_entries)} gold "
              f"— truncating gold to {len(predictions)}")
        gold_entries = gold_entries[:len(predictions)]

    # ── Iterate and collect failures ──────────────────────────────────────────
    counts   = Counter()
    failures = []

    for i, entry in enumerate(gold_entries):
        db_id    = entry.get('db_id', '')
        question = entry.get('question', '')
        gold_sql = entry.get('query', '')
        pred_sql = predictions[i]

        # Compute hardness from gold SQL keywords
        hardness = compute_hardness(gold_sql)

        # Hardness filter
        if args.hardness and hardness not in args.hardness:
            counts['skipped'] += 1
            continue

        counts['total'] += 1

        # Locate database
        db_path = os.path.join(args.db, db_id, f'{db_id}.sqlite')
        if not os.path.exists(db_path):
            db_path = os.path.join(args.db, f'{db_id}.sqlite')

        # Execute both queries
        gold_res, _     = execute_sql(gold_sql, db_path)
        pred_res, p_err = execute_sql(pred_sql, db_path)

        # Classify
        if p_err:
            fail_type = 'exec_error'
            error_msg = p_err[:120]
        elif pred_res != gold_res:
            fail_type = 'exec_mismatch'
            error_msg = f"gold={str(gold_res)[:70]} | pred={str(pred_res)[:70]}"
        elif not em_match(pred_sql, gold_sql):
            fail_type = 'em_only'
            error_msg = ''
        else:
            counts['pass'] += 1
            continue   # perfect match — skip

        counts[fail_type] += 1
        failures.append(dict(
            idx      = i,
            db_id    = db_id,
            question = question,
            hardness = hardness,
            fail_type= fail_type,
            gold_sql = gold_sql,
            pred_sql = pred_sql,
            error_msg= error_msg,
        ))

    # ── Write CSV ──────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ['idx', 'db_id', 'question', 'hardness', 'fail_type',
              'gold_sql', 'pred_sql', 'error_msg']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(failures)

    # ── Summary ────────────────────────────────────────────────────────────────
    t       = counts['total']
    n_pass  = counts['pass']
    n_fail  = len(failures)
    ex_fail = counts['exec_error'] + counts['exec_mismatch']
    ex_pass = t - ex_fail

    print(f"\n{'='*60}")
    print(f"SPIDER FAILURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total evaluated : {t}")
    print(f"  EX passed       : {ex_pass}  ({ex_pass/t*100:.1f}%)")
    print(f"  EX failed       : {ex_fail}  ({ex_fail/t*100:.1f}%)")
    print(f"  ├─ exec_error   : {counts['exec_error']}")
    print(f"  └─ exec_mismatch: {counts['exec_mismatch']}")
    print(f"  em_only (EX ok) : {counts['em_only']}")
    print(f"  EM passed       : {n_pass}  ({n_pass/t*100:.1f}%)")
    print(f"{'='*60}")
    print(f"\n✓ Failures saved → {args.output}")

    # ── Breakdown by hardness ─────────────────────────────────────────────────
    ex_by_hardness = Counter(
        r['hardness'] for r in failures if r['fail_type'] != 'em_only'
    )
    em_by_hardness = Counter(
        r['hardness'] for r in failures
    )

    print(f"\nEX failures by hardness:")
    for level in ['easy', 'medium', 'hard', 'extra']:
        cnt = ex_by_hardness.get(level, 0)
        if cnt:
            print(f"  {level:<8}: {cnt}")

    print(f"\nAll failures by hardness:")
    for level in ['easy', 'medium', 'hard', 'extra']:
        cnt = em_by_hardness.get(level, 0)
        if cnt:
            print(f"  {level:<8}: {cnt}")


if __name__ == '__main__':
    main()