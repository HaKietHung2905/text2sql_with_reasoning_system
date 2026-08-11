"""
Chạy: python3 check_count_regressions.py

Kiểm tra execution result của predicted SQL vs gold SQL cho đúng 98 dòng
COUNT(*) vừa regen, để tìm ra dòng nào bị sai kết quả sau khi normalize
COUNT(col) -> COUNT(*).
"""
import json
import sqlite3
import os

LINES = [23,24,25,34,35,36,37,82,83,91,94,95,108,109,110,114,115,116,117,130,131,
         142,143,148,149,150,151,162,177,179,197,233,266,279,284,285,289,312,314,
         315,343,369,370,371,378,394,399,406,408,409,420,478,479,487,504,521,526,
         527,531,541,552,568,569,642,643,688,695,696,738,741,742,744,758,759,819,
         820,824,845,846,849,861,862,889,890,909,911,912,931,932,933,934,936,937,
         955,981,1013,1023,1024]

GOLD_PATH = "data/raw/spider/dev.json"
PRED_PATH = "results/predictions_fullsystem_spider.tsv"
DB_DIR    = "data/raw/spider/database"

def run_query(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted(rows)
    except Exception as e:
        return f"ERROR: {e}"

def main():
    with open(GOLD_PATH) as f:
        gold_data = json.load(f)
    with open(PRED_PATH) as f:
        pred_rows = [l.rstrip("\n") for l in f]

    mismatches = []
    for ln in LINES:
        idx = ln - 1
        if idx >= len(gold_data) or idx >= len(pred_rows):
            continue
        gold_sql = gold_data[idx]["query"]
        db_id    = gold_data[idx]["db_id"]
        pred_sql, _, _ = pred_rows[idx].partition("\t")
        db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")

        gold_result = run_query(db_path, gold_sql)
        pred_result = run_query(db_path, pred_sql)

        if gold_result != pred_result:
            mismatches.append((ln, db_id, gold_sql, pred_sql, gold_result, pred_result))

    print(f"Tổng số dòng kiểm tra: {len(LINES)}")
    print(f"Số dòng execution KHÔNG khớp: {len(mismatches)}\n")
    for ln, db_id, g, p, gr, pr in mismatches:
        print(f"--- Line {ln} (db={db_id}) ---")
        print(f"  GOLD: {g}")
        print(f"  PRED: {p}")
        gr_s = str(gr)[:150]
        pr_s = str(pr)[:150]
        print(f"  gold_result: {gr_s}")
        print(f"  pred_result: {pr_s}")
        print()

if __name__ == "__main__":
    main()