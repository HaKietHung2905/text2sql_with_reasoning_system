#!/usr/bin/env python3
"""
WikiSQL Failure Analyser
========================
Reads predictions TSV + original dev.json and produces:
  1. results/wikisql_em_failures.csv   — every failure row, sorted by category
  2. results/wikisql_em_report.txt     — human-readable report with examples

Usage:
  python scripts/analyze_wikisql_failures.py \
      --gold    data/raw/wikisql/dev.json \
      --table   data/raw/wikisql/tables.json \
      --predict results/predictions_wikisql_v2.tsv \
      --out_dir results/failure_analysis
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Reuse the same structural EM helpers from evaluate_wikisql.py ─────────────

_SEM_AGG_OPS  = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_SEM_AGG_RE   = re.compile(r'\b(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*(?:DISTINCT\s+)?(.*?)\s*\)', re.IGNORECASE)
_SEM_SEL_RE   = re.compile(r'\bSELECT\s+(.+?)\s+FROM\b', re.IGNORECASE | re.DOTALL)
_SEM_WHERE_RE = re.compile(r'\bWHERE\s+(.+?)(?:\s+(?:ORDER|GROUP|HAVING|LIMIT)\b|$)', re.IGNORECASE | re.DOTALL)
_SEM_COND_RE  = re.compile(
    r'([\w\s\-/.()\[\]#&]+?)'
    r'\s*(=|!=|<>|>=|<=|>|<|LIKE|NOT\s+LIKE|NOT\s+IN|IN)\s*'
    r"((?:'(?:[^'\\]|\\.)*')|(?:\"[^\"]*\")|(?:\S+))",
    re.IGNORECASE)

_SEM_MONTH_MAP = {
    'january':'01','february':'02','march':'03','april':'04','may':'05','june':'06',
    'july':'07','august':'08','september':'09','october':'10','november':'11','december':'12',
    'jan':'01','feb':'02','mar':'03','apr':'04','jun':'06',
    'jul':'07','aug':'08','sep':'09','sept':'09','oct':'10','nov':'11','dec':'12',
}


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    return v

def _norm_col(name: str) -> str:
    result = re.sub(r"[\s_/.()\-,;:#&'\"!?%@$]+", " ", name.lower().strip()).strip()
    result = re.sub(r"(\d+)", r" \1 ", result)
    return re.sub(r"\s+", " ", result).strip()

def _col_index(name: str, headers: List[str]) -> Optional[int]:
    if not name: return None
    name = name.strip()
    m = re.match(r"^col(\d+)$", name, re.IGNORECASE)
    if m:
        idx = int(m.group(1)); return idx if idx < len(headers) else None
    for i, h in enumerate(headers):
        if h == name: return i
    norm = _norm_col(name)
    if norm:
        for i, h in enumerate(headers):
            if _norm_col(h) == norm: return i
    name_toks = set(norm.split())
    if name_toks:
        best_j, best_i = 0.0, None
        for i, h in enumerate(headers):
            h_toks = set(_norm_col(h).split())
            if not h_toks: continue
            j = len(name_toks & h_toks) / len(name_toks | h_toks)
            if j >= 0.6 and j > best_j: best_j, best_i = j, i
        if best_i is not None: return best_i
    norm_low = norm.lower()
    for i, h in enumerate(headers):
        h_norm = _norm_col(h).lower()
        if h_norm.startswith(norm_low) or norm_low.startswith(h_norm): return i
    return None

def _op_index(op: str) -> int:
    op = op.strip().upper()
    if op == "=": return 0
    if op in (">", ">="): return 1
    if op in ("<", "<="): return 2
    return 3

def _norm_val(v: Any) -> Any:
    if not isinstance(v, str):
        if isinstance(v, float) and v == int(v): return int(v)
        return v
    v = v.strip().rstrip("%").replace("\u2212", "-").replace("\u2013", "-")
    v = v.strip('"').strip("'")
    if v.lower().startswith("the "): v = v[4:]
    v_lower = v.lower()
    try: return float(v_lower) if "." in v_lower else int(v_lower)
    except (ValueError, TypeError): pass
    if re.search(r"[a-z]", v_lower): return re.sub(r"\s+", " ", v_lower).strip()
    return re.sub(r"[\s,]+", "", v_lower)

def _vals_match(v1: Any, v2: Any) -> bool:
    n1, n2 = _norm_val(v1), _norm_val(v2)
    if n1 == n2: return True
    try: return abs(float(n1) - float(n2)) < 1e-9
    except (ValueError, TypeError): pass
    if isinstance(n1, str) and isinstance(n2, str):
        s1, s2 = str(n1).lower(), str(n2).lower()
        return s1.startswith(s2) or s2.startswith(s1)
    return False

def _parse_val(raw: str) -> Any:
    v = _strip_quotes(raw).strip()
    v = v.replace("\u2212", "-").replace("\u2013", "-")
    v = v.strip('"').strip("'").rstrip("%").replace("\\'", "'").rstrip("\\")
    try: return float(v) if "." in v else int(v)
    except (ValueError, TypeError): return v.lower()

def parse_sql(sql: str, headers: List[str]) -> Optional[Dict]:
    if not sql or not sql.strip(): return None
    sql = sql.strip().rstrip(";")
    _CSTAR = -999
    agg_id = 0; sel_col = None; count_star = False
    m_agg = _SEM_AGG_RE.search(sql)
    if m_agg:
        agg_id = _SEM_AGG_OPS.index(m_agg.group(1).upper())
        col_name = m_agg.group(2).strip().strip("`\"[]")
        if col_name == "*": count_star = True
        else: sel_col = col_name
    else:
        m_sel = _SEM_SEL_RE.search(sql)
        if m_sel:
            sel_col = m_sel.group(1).strip().strip("`\"[]")
            if sel_col == "*": sel_col = headers[0] if headers else None
    if count_star: sel_idx = _CSTAR
    else:
        if sel_col is None: return None
        sel_idx = _col_index(sel_col, headers)
        if sel_idx is None: return None
    conds = []
    m_where = _SEM_WHERE_RE.search(sql)
    if m_where:
        wc = m_where.group(1)
        wc = re.sub(r"^\s*(AND|OR)\s+", "", wc, flags=re.IGNORECASE)
        wc = re.sub(r"\(\s*SELECT\s+.+?\)", "?", wc, flags=re.IGNORECASE|re.DOTALL)
        for col_raw, op_raw, val_raw in _SEM_COND_RE.findall(wc):
            col_raw = col_raw.strip().strip("`\"[]")
            col_raw = re.sub(r"^(?:AND|OR|NOT)\s+", "", col_raw, flags=re.IGNORECASE).strip()
            col_idx = _col_index(col_raw, headers)
            if col_idx is None: continue
            actual_op = op_raw.strip().upper()
            if actual_op == "LIKE":
                vc = _strip_quotes(val_raw)
                op_idx = 0 if "%" not in vc and "_" not in vc else _op_index(op_raw)
            else:
                op_idx = _op_index(op_raw)
            conds.append([col_idx, op_idx, _parse_val(val_raw)])
    return {"agg": agg_id, "sel": sel_idx, "conds": conds, "count_star": count_star}

def gold_to_struct(gold_sql: Any, headers: List[str]) -> Optional[Dict]:
    if isinstance(gold_sql, str): return parse_sql(gold_sql, headers)
    if not isinstance(gold_sql, dict): return None
    conds = []
    for cond in gold_sql.get("conds", []):
        if len(cond) < 3: continue
        col_idx, op_idx, value = cond[0], cond[1], cond[2]
        try:
            if isinstance(value, str):
                value = float(value) if "." in value else (
                    int(value) if value.lstrip("-").isdigit() else value)
        except (ValueError, TypeError): pass
        conds.append([col_idx, op_idx, value])
    agg = gold_sql.get("agg", 0)
    sel = gold_sql.get("sel", 0)
    return {"agg": agg, "sel": sel, "conds": conds,
            "count_star": (agg == _SEM_AGG_OPS.index("COUNT"))}

def load_headers_map(table_file: str) -> Dict[str, List[str]]:
    hmap: Dict[str, List[str]] = {}
    with open(table_file, encoding="utf-8") as f:
        tables = json.load(f)
    for t in tables:
        db_id = t.get("db_id") or t.get("id", "")
        col_names = t.get("column_names_original") or t.get("column_names", [])
        if col_names:
            headers = [c[1] for c in col_names if isinstance(c, list) and len(c)>=2 and c[0]==0]
        else:
            headers = t.get("header", [])
        if db_id and headers:
            hmap[db_id] = headers
    return hmap

# ── COND sub-category classifier ──────────────────────────────────────────────

def classify_cond_failure(pred_conds: List, gold_conds: List, headers: List[str]) -> str:
    """Classify WHY the COND comparison failed."""
    # Missing condition: pred has fewer than gold
    if len(pred_conds) < len(gold_conds):
        missing = len(gold_conds) - len(pred_conds)
        return f"missing_cond({missing} of {len(gold_conds)} absent)"

    # Extra but wrong: pred has enough but values don't match
    # Try to find which gold cond failed and why
    reasons = []
    matched_pred = [False] * len(pred_conds)
    for gc in gold_conds:
        gc_val  = _norm_val(gc[2])
        gc_col  = gc[0]
        gc_op   = gc[1]
        col_name = headers[gc_col] if gc_col < len(headers) else f"col{gc_col}"

        # Check: does any pred have matching col + val?
        col_val_match = any(
            pc[0] == gc_col and _vals_match(pc[2], gc[2])
            for i, pc in enumerate(pred_conds) if not matched_pred[i]
        )
        # Check: does any pred have matching val (any col)?
        val_match_any_col = any(
            _vals_match(pc[2], gc[2])
            for i, pc in enumerate(pred_conds) if not matched_pred[i]
        )
        # Check: does any pred have matching col (wrong val)?
        col_match_wrong_val = any(
            pc[0] == gc_col and not _vals_match(pc[2], gc[2])
            for i, pc in enumerate(pred_conds) if not matched_pred[i]
        )

        if col_val_match:
            # Matched col+val but op still wrong (shouldn't happen after Pass 3)
            reasons.append(f"op_mismatch({col_name})")
        elif col_match_wrong_val:
            # Right column, wrong value
            pred_val = next(
                pc[2] for i, pc in enumerate(pred_conds)
                if not matched_pred[i] and pc[0] == gc_col)
            reasons.append(f"wrong_val({col_name}: gold={repr(gc[2])[:30]} pred={repr(pred_val)[:30]})")
        elif val_match_any_col:
            reasons.append(f"wrong_col(val={repr(gc[2])[:25]} in wrong column)")
        else:
            reasons.append(f"no_match({col_name}='{repr(gc[2])[:20]}')")

    return " | ".join(reasons) if reasons else "unknown"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WikiSQL Failure Analyser")
    parser.add_argument("--gold",    required=True, help="data/raw/wikisql/dev.json")
    parser.add_argument("--table",   required=True, help="data/raw/wikisql/tables.json")
    parser.add_argument("--predict", required=True, help="results/predictions_wikisql_v2.tsv")
    parser.add_argument("--out_dir", default="results/failure_analysis")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.gold, encoding="utf-8") as f:
        gold_data = json.load(f)
    preds: List[str] = []
    with open(args.predict, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            preds.append(line.split("\t")[0].strip() if "\t" in line else line.strip())

    effective_limit = min(args.limit, len(preds)) if args.limit else len(preds)
    gold_data = gold_data[:effective_limit]
    preds     = preds[:effective_limit]

    # ── Build headers map ─────────────────────────────────────────────────────
    hmap = load_headers_map(args.table)
    for item in gold_data:
        db_id = item.get("db_id") or item.get("table_id", "")
        if db_id and db_id not in hmap:
            tbl = item.get("table", {})
            hdrs = tbl.get("header", []) if isinstance(tbl, dict) else []
            if hdrs: hmap[db_id] = hdrs

    # ── Evaluate ──────────────────────────────────────────────────────────────
    _MAX = _SEM_AGG_OPS.index("MAX")
    _MIN = _SEM_AGG_OPS.index("MIN")

    rows: List[Dict] = []
    total = correct = 0
    fail_cats: Counter = Counter()
    cond_sub_cats: Counter = Counter()

    for gold_item, pred_sql in zip(gold_data, preds):
        db_id    = gold_item.get("db_id") or gold_item.get("table_id", "")
        question = gold_item.get("question", "")
        headers  = hmap.get(db_id, [])
        gold_sql = gold_item.get("sql", {})

        gold_struct = gold_to_struct(gold_sql, headers)
        pred_struct = parse_sql(pred_sql, headers)
        total += 1

        agg_ok = sel_ok = cond_ok = False
        fail_cat = ""
        cond_detail = ""

        if gold_struct is None:
            fail_cat = "parse_fail_gold"
        elif pred_struct is None:
            fail_cat = "parse_fail_pred"
        else:
            # AGG (with MAX/MIN relaxation)
            agg_ok = (pred_struct["agg"] == gold_struct["agg"])
            if not agg_ok:
                if pred_struct["agg"] == 0 and gold_struct["agg"] in (_MAX, _MIN) and gold_struct["conds"]:
                    agg_ok = True

            # SEL
            if pred_struct.get("count_star") and pred_struct["agg"] == gold_struct["agg"] == _SEM_AGG_OPS.index("COUNT"):
                sel_ok = True
            else:
                sel_ok = (pred_struct["sel"] == gold_struct["sel"])

            # COND — multi-pass
            pc = pred_struct["conds"]
            gc = gold_struct["conds"]
            if len(pc) >= len(gc):
                def _norm_c(c):
                    val = _norm_val(c[2])
                    if isinstance(val, str):
                        try: val = float(val) if "." in val else int(val)
                        except (ValueError, TypeError): pass
                    return (c[0], c[1], val)
                gn = [_norm_c(c) for c in gc]
                pn = [_norm_c(c) for c in pc]
                mp = [False] * len(pn)
                all_found = True
                for gcn in gn:
                    found = False
                    for i, pcn in enumerate(pn):
                        if mp[i]: continue
                        if pcn[0]==gcn[0] and pcn[1]==gcn[1] and _vals_match(pcn[2],gcn[2]):
                            mp[i]=True; found=True; break
                    if not found:
                        for i, pcn in enumerate(pn):
                            if mp[i]: continue
                            if pcn[1]==gcn[1] and _vals_match(pcn[2],gcn[2]):
                                mp[i]=True; found=True; break
                    if not found:
                        for i, pcn in enumerate(pn):
                            if mp[i]: continue
                            if pcn[0]==gcn[0] and _vals_match(pcn[2],gcn[2]):
                                mp[i]=True; found=True; break
                    if not found:
                        all_found = False; break
                cond_ok = all_found
            # else: len(pc) < len(gc) → cond_ok stays False

            if not cond_ok:
                cond_detail = classify_cond_failure(pc, gc, headers)

            parts = []
            if not agg_ok:  parts.append("agg")
            if not sel_ok:  parts.append("sel")
            if not cond_ok: parts.append("cond")
            fail_cat = "+".join(parts) if parts else "ok"

        if fail_cat == "ok":
            correct += 1
        fail_cats[fail_cat] += 1

        # Decode gold struct for readability
        gold_agg_str = _SEM_AGG_OPS[gold_struct["agg"]] if gold_struct else "?"
        gold_sel_str = (headers[gold_struct["sel"]] if gold_struct and gold_struct["sel"] >= 0
                        and gold_struct["sel"] < len(headers) else "?")
        gold_conds_str = ""
        if gold_struct:
            parts_c = []
            for c in gold_struct["conds"]:
                col_name = headers[c[0]] if c[0] < len(headers) else f"col{c[0]}"
                op_str = ["=",">","<","OP"][c[1]] if c[1] < 4 else str(c[1])
                parts_c.append(f"{col_name} {op_str} '{c[2]}'")
            gold_conds_str = " AND ".join(parts_c)

        pred_agg_str = _SEM_AGG_OPS[pred_struct["agg"]] if pred_struct else "?"
        pred_sel_str = (headers[pred_struct["sel"]] if pred_struct and pred_struct.get("sel", -999) >= 0
                        and pred_struct["sel"] < len(headers) else "?")
        pred_conds_str = ""
        if pred_struct:
            parts_c = []
            for c in pred_struct["conds"]:
                col_name = headers[c[0]] if c[0] < len(headers) else f"col{c[0]}"
                op_str = ["=",">","<","OP"][c[1]] if c[1] < 4 else str(c[1])
                parts_c.append(f"{col_name} {op_str} '{c[2]}'")
            pred_conds_str = " AND ".join(parts_c)

        rows.append({
            "line_no"      : total,
            "db_id"        : db_id,
            "em"           : fail_cat == "ok",
            "fail_cat"     : fail_cat,
            "agg_ok"       : agg_ok,
            "sel_ok"       : sel_ok,
            "cond_ok"      : cond_ok,
            "question"     : question,
            "pred_sql"     : pred_sql,
            "gold_agg"     : gold_agg_str,
            "pred_agg"     : pred_agg_str,
            "gold_sel"     : gold_sel_str,
            "pred_sel"     : pred_sel_str,
            "gold_conds"   : gold_conds_str,
            "pred_conds"   : pred_conds_str,
            "cond_detail"  : cond_detail,
            "gold_sql_raw" : str(gold_sql),
        })

        # Track cond sub-categories
        if "cond" in fail_cat and cond_detail:
            sub = cond_detail.split("(")[0]
            cond_sub_cats[sub] += 1

    struct_em = correct / total if total else 0

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "em_failures.csv"
    failure_rows = [r for r in rows if not r["em"]]
    failure_rows.sort(key=lambda r: r["fail_cat"])

    fieldnames = [
        "line_no","fail_cat","agg_ok","sel_ok","cond_ok",
        "question","pred_sql","pred_agg","pred_sel","pred_conds",
        "gold_agg","gold_sel","gold_conds","cond_detail","db_id","gold_sql_raw",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failure_rows)
    print(f"✓ CSV  → {csv_path}  ({len(failure_rows)} failure rows)")

    # ── Save human-readable report ─────────────────────────────────────────────
    txt_path = out_dir / "em_report.txt"
    n_fail = len(failure_rows)
    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("  WikiSQL Structural EM Failure Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  Total evaluated   : {total}\n")
        f.write(f"  Structural EM     : {struct_em:.2%}  ({correct}/{total})\n")
        f.write(f"  Total failures    : {n_fail}\n\n")

        # Failure category table
        f.write("── FAILURE CATEGORY BREAKDOWN ─────────────────────────────────────────\n\n")
        f.write(f"  {'Category':<22} {'Count':>6}  {'% of total':>10}  {'% of fails':>11}\n")
        f.write(f"  {'─'*22} {'─'*6}  {'─'*10}  {'─'*11}\n")
        ORDER = ["parse_fail_pred","parse_fail_gold","agg","sel","cond",
                 "agg+sel","agg+cond","sel+cond","agg+sel+cond"]
        for cat in ORDER + [k for k in fail_cats if k not in ORDER and k != "ok"]:
            cnt = fail_cats.get(cat, 0)
            if cnt == 0: continue
            f.write(f"  {cat:<22} {cnt:>6}  {cnt/total*100:>9.1f}%  {cnt/n_fail*100:>10.1f}%\n")

        # COND sub-category breakdown
        if cond_sub_cats:
            f.write("\n── COND FAILURE SUB-CATEGORIES ────────────────────────────────────────\n\n")
            f.write(f"  {'Sub-category':<30} {'Count':>6}  {'% of cond fails':>16}\n")
            f.write(f"  {'─'*30} {'─'*6}  {'─'*16}\n")
            total_cond_fails = sum(cond_sub_cats.values())
            for sub, cnt in cond_sub_cats.most_common():
                f.write(f"  {sub:<30} {cnt:>6}  {cnt/total_cond_fails*100:>15.1f}%\n")

        # ── Per-category examples ──────────────────────────────────────────────
        N_EXAMPLES = 15
        cats_to_show = [c for c in ORDER if fail_cats.get(c, 0) > 0]

        for cat in cats_to_show:
            examples = [r for r in failure_rows if r["fail_cat"] == cat][:N_EXAMPLES]
            if not examples: continue
            f.write(f"\n\n{'═'*80}\n")
            f.write(f"  CATEGORY: {cat}  ({fail_cats[cat]} total failures)\n")
            f.write(f"{'═'*80}\n")

            for i, r in enumerate(examples, 1):
                f.write(f"\n  [{i}] Line {r['line_no']}  db={r['db_id']}\n")
                f.write(f"  Q  : {r['question']}\n")
                f.write(f"  SQL: {r['pred_sql'][:120]}\n")
                if cat in ("agg", "agg+sel", "agg+cond", "agg+sel+cond"):
                    f.write(f"  AGG: gold={r['gold_agg']!r:<6}  pred={r['pred_agg']!r}\n")
                if cat in ("sel", "agg+sel", "sel+cond", "agg+sel+cond"):
                    f.write(f"  SEL: gold={r['gold_sel']!r:<20}  pred={r['pred_sel']!r}\n")
                if "cond" in cat:
                    f.write(f"  COND gold: {r['gold_conds']}\n")
                    f.write(f"  COND pred: {r['pred_conds']}\n")
                    if r["cond_detail"]:
                        f.write(f"  WHY:       {r['cond_detail']}\n")
                f.write(f"  {'─'*70}\n")

    print(f"✓ Report → {txt_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FAILURE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Structural EM   : {struct_em:.2%}  ({correct}/{total})")
    print(f"  Total failures  : {n_fail}")
    print()
    print(f"  {'Category':<22} {'Count':>6}  {'% of fails':>11}")
    print(f"  {'─'*22} {'─'*6}  {'─'*11}")
    for cat in ORDER + [k for k in fail_cats if k not in ORDER and k != "ok"]:
        cnt = fail_cats.get(cat, 0)
        if cnt == 0: continue
        print(f"  {cat:<22} {cnt:>6}  {cnt/n_fail*100:>10.1f}%")

    if cond_sub_cats:
        print()
        print(f"  COND sub-categories:")
        total_cond = sum(cond_sub_cats.values())
        for sub, cnt in cond_sub_cats.most_common(8):
            print(f"    {sub:<28} {cnt:>5}  ({cnt/total_cond*100:.1f}%)")

    print("=" * 70)
    print()
    print(f"  Open {txt_path} for 15 examples per failure category.")
    print(f"  Open {csv_path} in Excel/Sheets for full sortable table.")


if __name__ == "__main__":
    main()