#!/usr/bin/env python3
"""
extract_spider_failures.py — Extract all Spider evaluation failures to CSV
==========================================================================

Loads gold (dev.json) + predictions (.tsv), runs execution comparison,
and writes a failures CSV with: idx, db_id, question, hardness,
gold_sql, pred_sql, fail_type, error_msg.

fail_type values:
  exec_error   — predicted SQL raised an exception
  exec_mismatch — SQL executed but results differ from gold
  em_only       — execution passed but exact match failed

Usage:
  python extract_spider_failures.py \
      --gold    data/raw/spider/dev.json \
      --db      data/raw/spider/database \
      --predict results/predictions_spider.tsv \
      --output  results/spider_failures.csv

Optional:
  --hardness  hard extra      # filter to specific hardness levels
  --limit     200             # only process first N examples
"""

import sys
import os
import json
import sqlite3
import argparse
import csv
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# SQL execution helper
# ─────────────────────────────────────────────────────────────────────────────

def execute_sql(sql: str, db_path: str):
    """Execute SQL and return (results, error_msg). results=None on error."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        # Normalise: sort rows, strip whitespace from strings
        results = sorted([
            tuple(c.strip() if isinstance(c, str) else c for c in row)
            for row in results
        ])
        return results, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Hardness detection (lightweight, no Spider parser needed)
# ─────────────────────────────────────────────────────────────────────────────

def get_hardness_from_gold_entry(entry: dict) -> str:
    """Use pre-computed hardness if available, else 'unknown'."""
    return entry.get("hardness", "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Exact-match (string-level normalisation, not Spider parser)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(sql: str) -> str:
    sql = sql.lower().strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql


def em_match(pred: str, gold: str) -> bool:
    return _norm(pred) == _norm(gold)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Spider evaluation failures")
    parser.add_argument("--gold",    required=True, help="Path to dev.json")
    parser.add_argument("--db",      required=True, help="Path to database directory")
    parser.add_argument("--predict", required=True, help="Path to predictions .tsv")
    parser.add_argument("--output",  default="results/spider_failures.csv")
    parser.add_argument("--hardness", nargs="*",
                        choices=["easy", "medium", "hard", "extra", "unknown"],
                        default=None,
                        help="Filter to these hardness levels (default: all)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # ── Load gold ────────────────────────────────────────────────────────────
    with open(args.gold) as f:
        gold_entries = json.load(f)
    if args.limit:
        gold_entries = gold_entries[:args.limit]
    print(f"Loaded {len(gold_entries)} gold entries")

    # ── Load predictions ─────────────────────────────────────────────────────
    predictions = []
    with open(args.predict) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            pred_sql = parts[0].strip()
            predictions.append(pred_sql)

    if len(predictions) < len(gold_entries):
        print(f"WARNING: {len(predictions)} predictions vs {len(gold_entries)} gold — "
              f"truncating gold to {len(predictions)}")
        gold_entries = gold_entries[:len(predictions)]

    # ── Iterate and collect failures ─────────────────────────────────────────
    failures = []
    counts = {"total": 0, "exec_error": 0, "exec_mismatch": 0, "em_only": 0, "pass": 0}

    for i, entry in enumerate(gold_entries):
        db_id    = entry.get("db_id", "")
        question = entry.get("question", "")
        gold_sql = entry.get("query", "")
        hardness = get_hardness_from_gold_entry(entry)
        pred_sql = predictions[i]

        # Hardness filter
        if args.hardness and hardness not in args.hardness:
            continue

        counts["total"] += 1
        db_path = os.path.join(args.db, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            # Try flat layout
            db_path = os.path.join(args.db, f"{db_id}.sqlite")

        # ── Execute both ──────────────────────────────────────────────────
        gold_res, gold_err = execute_sql(gold_sql, db_path)
        pred_res, pred_err = execute_sql(pred_sql, db_path)

        # Determine fail type
        fail_type = None
        error_msg = ""

        if pred_err:
            fail_type = "exec_error"
            error_msg = pred_err[:120]
        elif pred_res != gold_res:
            fail_type = "exec_mismatch"
            error_msg = (
                f"gold={str(gold_res)[:80]} | pred={str(pred_res)[:80]}"
            )
        elif not em_match(pred_sql, gold_sql):
            fail_type = "em_only"   # execution passed, EM failed
        else:
            counts["pass"] += 1
            continue  # perfect — skip

        counts[fail_type] += 1
        failures.append({
            "idx":       i,
            "db_id":     db_id,
            "question":  question,
            "hardness":  hardness,
            "fail_type": fail_type,
            "gold_sql":  gold_sql,
            "pred_sql":  pred_sql,
            "error_msg": error_msg,
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ["idx", "db_id", "question", "hardness", "fail_type",
              "gold_sql", "pred_sql", "error_msg"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = counts["total"]
    n_fail  = len(failures)
    n_pass  = counts["pass"]

    print(f"\n{'='*60}")
    print(f"SPIDER FAILURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total evaluated : {total}")
    print(f"  Passed          : {n_pass}  ({n_pass/total*100:.1f}%)")
    print(f"  Failed          : {n_fail}  ({n_fail/total*100:.1f}%)")
    print(f"  ├─ exec_error   : {counts['exec_error']}")
    print(f"  ├─ exec_mismatch: {counts['exec_mismatch']}")
    print(f"  └─ em_only      : {counts['em_only']}")
    print(f"{'='*60}")
    print(f"\n✓ Failures saved → {args.output}")

    # ── Hardness breakdown of failures ───────────────────────────────────────
    from collections import Counter
    hcount = Counter(r["hardness"] for r in failures)
    print(f"\nFailures by hardness:")
    for level in ["easy", "medium", "hard", "extra", "unknown"]:
        if level in hcount:
            print(f"  {level:<8}: {hcount[level]}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
extract_spider_failures.py — Extract Spider evaluation failures to CSV
"""
import sys, os, json, sqlite3, argparse, csv, re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

# ── Hardness from SQL keywords ────────────────────────────────────────────────
def compute_hardness(sql: str) -> str:
    s = sql.lower()
    n_join     = len(re.findall(r'\bjoin\b', s))
    n_subquery = len(re.findall(r'\bselect\b', s)) - 1
    has_group  = bool(re.search(r'\bgroup\s+by\b', s))
    has_having = bool(re.search(r'\bhaving\b', s))
    has_set_op = bool(re.search(r'\b(intersect|except|union)\b', s))
    has_order  = bool(re.search(r'\border\s+by\b', s))
    has_limit  = bool(re.search(r'\blimit\b', s))
    n_cond     = len(re.findall(r'\b(and|or)\b', s))

    if has_set_op or n_subquery >= 2 or (n_join >= 2 and has_having):
        return 'extra'
    if n_subquery == 1 or (n_join >= 2) or (has_group and n_join >= 1):
        return 'hard'
    if has_group or has_having or n_join == 1 or n_cond >= 2:
        return 'medium'
    return 'easy'

# ── SQL execution ─────────────────────────────────────────────────────────────
def _coerce(v):
    """Normalise a result cell: strip strings, round floats."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 4)
    return v

def execute_sql(sql: str, db_path: str):
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

# ── String-level EM ───────────────────────────────────────────────────────────
def _norm(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql.lower().strip().rstrip(';'))

def em_match(pred, gold):
    return _norm(pred) == _norm(gold)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold',    required=True)
    ap.add_argument('--db',      required=True)
    ap.add_argument('--predict', required=True)
    ap.add_argument('--output',  default='results/spider_failures.csv')
    ap.add_argument('--hardness', nargs='*',
                    choices=['easy','medium','hard','extra'], default=None)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    with open(args.gold) as f:
        gold_entries = json.load(f)
    if args.limit:
        gold_entries = gold_entries[:args.limit]

    predictions = []
    with open(args.predict) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(line.split('\t')[0].strip())

    if len(predictions) < len(gold_entries):
        print(f"WARNING: {len(predictions)} predictions vs {len(gold_entries)} gold — truncating")
        gold_entries = gold_entries[:len(predictions)]

    counts = Counter()
    failures = []

    for i, entry in enumerate(gold_entries):
        db_id    = entry.get('db_id', '')
        question = entry.get('question', '')
        gold_sql = entry.get('query', '')
        pred_sql = predictions[i]
        hardness = compute_hardness(gold_sql)

        if args.hardness and hardness not in args.hardness:
            counts['skipped'] += 1
            continue

        counts['total'] += 1

        db_path = os.path.join(args.db, db_id, f'{db_id}.sqlite')
        if not os.path.exists(db_path):
            db_path = os.path.join(args.db, f'{db_id}.sqlite')

        gold_res, _     = execute_sql(gold_sql, db_path)
        pred_res, p_err = execute_sql(pred_sql, db_path)

        if p_err:
            fail_type = 'exec_error'
            error_msg = p_err[:120]
        elif pred_res != gold_res:
            fail_type = 'exec_mismatch'
            error_msg = f"gold={str(gold_res)[:60]} | pred={str(pred_res)[:60]}"
        elif not em_match(pred_sql, gold_sql):
            fail_type = 'em_only'
            error_msg = ''
        else:
            counts['pass'] += 1
            continue

        counts[fail_type] += 1
        failures.append(dict(
            idx=i, db_id=db_id, question=question, hardness=hardness,
            fail_type=fail_type, gold_sql=gold_sql, pred_sql=pred_sql,
            error_msg=error_msg,
        ))

    # ── Write CSV ──────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = ['idx','db_id','question','hardness','fail_type','gold_sql','pred_sql','error_msg']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(failures)

    # ── Summary ────────────────────────────────────────────────────────────────
    t = counts['total']
    p = counts['pass']
    n = len(failures)
    ex_fail = counts['exec_error'] + counts['exec_mismatch']

    print(f"\n{'='*60}")
    print(f"SPIDER FAILURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total evaluated : {t}")
    print(f"  EX passed       : {t - ex_fail}  ({(t-ex_fail)/t*100:.1f}%)")
    print(f"  EX failed       : {ex_fail}  ({ex_fail/t*100:.1f}%)")
    print(f"  ├─ exec_error   : {counts['exec_error']}")
    print(f"  └─ exec_mismatch: {counts['exec_mismatch']}")
    print(f"  em_only (EX ok) : {counts['em_only']}")
    print(f"{'='*60}")
    print(f"\n✓ All failures saved → {args.output}")

    hcount = Counter(r['hardness'] for r in failures if r['fail_type'] != 'em_only')
    print(f"\nEX failures by hardness:")
    for level in ['easy','medium','hard','extra']:
        if level in hcount:
            print(f"  {level:<8}: {hcount[level]}")

if __name__ == '__main__':
    main()