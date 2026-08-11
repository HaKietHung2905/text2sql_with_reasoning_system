#!/usr/bin/env python3
"""
regen_failed.py — Re-generate failed predictions in-place
==========================================================

"""

import sys, os, json, shutil, argparse, logging, warnings, time, re, subprocess
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

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
    """Lines whose SQL is exactly the fallback placeholder used when
    generation failed for technical reasons (see regen_one / SQLGenerator)."""
    return [i+1 for i, row in enumerate(rows)
            if row.split("\t")[0].strip().upper() == marker.upper()]


# ─── Mode : Standard regeneration (pipeline-based) ───────────────────────
def init_pipelines(args):
    from utils.sql_schema import load_full_db_context  # noqa
    semantic_pipeline = reasoning_pipeline = retriever = None

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

    # ── Semantic RAG retriever (was completely missing before) ──────────────
    if args.use_chromadb:
        try:
            from src.retrieval.retriever import SpiderRetriever
            prefix = "wikisql" if ("wikisql" in args.questions.lower()
                                    or "wikisql" in args.db.lower()) else "spider"
            retriever = SpiderRetriever(
                persist_dir=args.chromadb_persist_dir, collection_prefix=prefix)
            print(f"✓ Semantic RAG retriever ready (prefix={prefix}, top_k={args.top_k})")
        except Exception as e:
            print(f"⚠ Semantic RAG retriever failed: {e}")

    try:
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        print("✓ SQLGenerator ready")
    except Exception as e:
        print(f"✗ SQLGenerator failed: {e}")
        sys.exit(1)

    return semantic_pipeline, reasoning_pipeline, sql_generator, retriever

def regen_one(line_no, questions_data, db_dir,
              semantic_pipeline, reasoning_pipeline, sql_generator,
              retriever=None, top_k=3, max_retries=3, current_row=None):
    from utils.sql_schema import load_full_db_context
    idx = line_no - 1
    if idx < 0 or idx >= len(questions_data):
        fallback_db_id = current_row.split("\t", 1)[1] if current_row and "\t" in current_row else ""
        return {"line": line_no, "db_id": fallback_db_id, "question": "",
                "sql": "SELECT 1", "tsv_row": f"SELECT 1\t{fallback_db_id}",
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
            semantic_hints = None
            if semantic_pipeline:
                try:
                    res = semantic_pipeline.enhance_question(question, db_id, None)
                    enhanced = res.get("enhanced_question", question)
                    semantic_hints = res.get("semantic_hints") or None
                except Exception:
                    pass

            few_shot_examples = None
            if retriever:
                try:
                    rag_result = retriever.retrieve_similar_questions(enhanced, n_results=top_k)
                    few_shot_examples = rag_result.get("results")
                except Exception:
                    few_shot_examples = None

            sql = ""
            if reasoning_pipeline:
                try:
                    ctx = load_full_db_context(db_id, db_dir)
                    rb  = reasoning_pipeline.generate_with_reasoning(
                        question=enhanced, db_id=db_id,
                        schema=ctx.get("schema", {}),
                        gold_sql=item.get("query", item.get("sql")),
                        db_path=db_path,
                        sql_generator=lambda q, strategy_hints=None: sql_generator.generate(
                            q, db_path,
                            few_shot_examples=few_shot_examples,
                            semantic_hints=semantic_hints,
                            strategy_hints=strategy_hints))
                    sql = rb.get("sql", "") or ""
                except Exception as e:
                    last_error = str(e)

            if not sql or sql.strip().upper() == "SELECT 1":
                sql = sql_generator.generate(
                    enhanced, db_path,
                    few_shot_examples=few_shot_examples,
                    semantic_hints=semantic_hints)

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    # Paths
    parser.add_argument("--predict",   required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--db",        required=True)

    # Which lines (Mode 4: explicit list; default: auto-detect SELECT 1)
    parser.add_argument("--lines", type=int, nargs="+", metavar="N",
                        help="Explicit line numbers to regenerate (Mode 4). "
                             "If omitted, auto-detects lines whose SQL is "
                             "exactly 'SELECT 1' (Mode 1).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print target lines without calling the API.")

    # Pipeline flags
    parser.add_argument("--use_reasoning_bank", action="store_true")
    parser.add_argument("--use_chromadb",       action="store_true")
    parser.add_argument("--chromadb_persist_dir", default="./data/embeddings/chroma_db")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--use_semantic",       action="store_true")
    parser.add_argument("--reasoning_config",   default="configs/reasoning_config.yaml")
    parser.add_argument("--semantic_config",    default=None)

    # Safety / resume
    parser.add_argument("--max_retries", type=int,   default=3)
    parser.add_argument("--delay",       type=float, default=0.4)
    parser.add_argument("--no_backup",   action="store_true")

    args = parser.parse_args()

    if not Path(args.predict).exists():
        print(f"✗ File not found: {args.predict}"); sys.exit(1)

    rows = load_tsv(args.predict)
    print(f"✓ Loaded {len(rows)} rows from {args.predict}")

    with open(args.questions, encoding="utf-8") as f:
        questions_data = json.load(f)
    questions = [item.get("question", "") for item in questions_data]
    print(f"✓ Loaded {len(questions)} questions")

    if args.lines:
        target_lines = sorted(set(args.lines))
        print(f"  Mode: specific lines → {len(target_lines)}")
    else:
        target_lines = find_failed_lines(rows)
        print(f"  Mode: auto SELECT 1 (connection/generation failures) → {len(target_lines)} found")

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
    sem, rb, sg, retriever = init_pipelines(args)
    print()

    fixed = still_failed = 0
    for ln in target_lines:
        if ln > len(rows):
            print(f"⚠ Line {ln} beyond TSV — skipping"); continue
        r = regen_one(ln, questions_data, args.db, sem, rb, sg,
                    retriever=retriever, top_k=args.top_k,
                    max_retries=args.max_retries, current_row=rows[ln-1])
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


if __name__ == "__main__":
    main()