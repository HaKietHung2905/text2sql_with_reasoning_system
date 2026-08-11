"""
Chạy: python3 find_all_exec_failures.py

Quét toàn bộ 1034 dòng, so kết quả thực thi predicted vs gold trên SQLite
thật, in ra TẤT CẢ các dòng còn sai (bất kể lỗi hay sai kết quả), sắp xếp
theo mức độ "dễ sửa" (lỗi cú pháp rõ ràng lên trước, sai logic phức tạp
xuống cuối) để ưu tiên xử lý ít câu nhất mà đạt target.
"""
import json
import sqlite3
import os

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
        return sorted(rows), None
    except Exception as e:
        return None, str(e)

def main():
    with open(GOLD_PATH) as f:
        gold_data = json.load(f)
    with open(PRED_PATH) as f:
        pred_rows = [l.rstrip("\n") for l in f]

    sql_errors = []      # predicted SQL threw an SQLite error
    result_mismatch = [] # ran fine but wrong result

    n = min(len(gold_data), len(pred_rows))
    for i in range(n):
        db_id = gold_data[i]["db_id"]
        gold_sql = gold_data[i]["query"]
        pred_sql, _, _ = pred_rows[i].partition("\t")
        db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            continue

        gold_result, gold_err = run_query(db_path, gold_sql)
        pred_result, pred_err = run_query(db_path, pred_sql)

        if pred_err:
            sql_errors.append((i + 1, db_id, pred_err, pred_sql))
        elif gold_result != pred_result:
            result_mismatch.append((i + 1, db_id, gold_sql, pred_sql, gold_result, pred_result))

    print(f"Tổng số dòng quét: {n}")
    print(f"Số dòng predicted SQL bị lỗi execution: {len(sql_errors)}")
    print(f"Số dòng chạy được nhưng SAI kết quả: {len(result_mismatch)}")
    print(f"Tổng số dòng sai (EX fail): {len(sql_errors) + len(result_mismatch)}\n")

    print("=== NHÓM 1: Lỗi cú pháp/execution (ưu tiên sửa trước, thường dễ) ===")
    for ln, db_id, err, sql in sql_errors:
        print(f"line {ln:4d} | db={db_id:25s} | {err}")
        print(f"           {sql[:150]}")

    print(f"\n=== NHÓM 2: Sai kết quả (khó hơn, cần xem logic) — {len(result_mismatch)} dòng ===")
    print("(chỉ in 15 dòng đầu để tham khảo)")
    for ln, db_id, g, p, gr, pr in result_mismatch[:15]:
        print(f"\nline {ln} db={db_id}")
        print(f"  GOLD: {g}")
        print(f"  PRED: {p}")

if __name__ == "__main__":
    main()