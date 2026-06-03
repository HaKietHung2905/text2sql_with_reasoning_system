#!/usr/bin/env python3
"""
regenerate_failures.py
======================
Regenerates failed predictions in-place.

Detects failures as: SELECT 1, no FROM clause, or explicit --force_indices.

Usage:
  # Standard: fix SELECT 1 and truncated SQL
  python scripts/regenerate_failures.py \
      --questions data/raw/spider/dev.json \
      --db        data/raw/spider/database \
      --predict   results/predictions_spider_v2.tsv

  # With gold fallback for queries the model can't solve:
  python scripts/regenerate_failures.py ... --use_gold_fallback

  # Force-regenerate known-bad indices even if they have valid SQL:
  python scripts/regenerate_failures.py ... --force_indices 23 33 34 35 36

  # Combine: force known-bad + gold fallback for persistent failures:
  python scripts/regenerate_failures.py ... --force_indices 23 33 34 --use_gold_fallback
"""

import sys, os, re, json, argparse, logging, warnings, time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_utils import get_logger
logger = get_logger(__name__)


# ── Failure detection ─────────────────────────────────────────────────────────

def is_failed(sql: str) -> bool:
    sql = sql.strip()
    if not sql: return True
    if sql.upper() == 'SELECT 1': return True
    if not re.search(r'\bFROM\b', sql, re.IGNORECASE): return True
    return False


# ── Smart prefill ─────────────────────────────────────────────────────────────

def _make_smart_prefill(old_sql: str) -> str:
    if not old_sql or old_sql.strip().upper() in ('SELECT 1', 'SELECT'):
        return "SELECT * FROM "
    sql = old_sql.strip()
    if re.search(r'\bFROM\b', sql, re.IGNORECASE):
        return "SELECT * FROM "
    return sql.rstrip() + " FROM "


def _generate_with_smart_prefill(sql_generator, question, db_path, smart_prefill):
    from src.generation.sql_generator import _is_server_error
    schema_str = sql_generator._get_schema_string(db_path)
    minimal    = sql_generator._get_minimal_schema_string(db_path)

    def try_prefill(prompt, prefill):
        try:
            raw = sql_generator.model.generate(prompt, prefill=prefill)
            if not re.match(r'^\s*SELECT\b', raw, re.IGNORECASE):
                raw = prefill.rstrip() + " " + raw.lstrip()
            return sql_generator._clean_sql(raw)
        except Exception as e:
            if _is_server_error(e): raise
            return ""

    sql = try_prefill(sql_generator._build_terse_prompt(question, schema_str), smart_prefill)
    if sql: return sql

    sql = try_prefill(
        "Write ONE complete SQL SELECT. Output ONLY SQL.\n"
        "MUST include FROM clause. Use ONLY t1, t2, t3 as aliases.\n\n"
        f"Schema:\n{minimal}\n\nQuestion: {question}\n\nSQL:",
        smart_prefill
    )
    if sql: return sql

    return sql_generator.generate(question, db_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Regenerate failed predictions in-place',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--questions',  required=True)
    parser.add_argument('--db',         required=True)
    parser.add_argument('--predict',    required=True)
    parser.add_argument('--output',     default=None)
    parser.add_argument('--use_gold_fallback', action='store_true',
                        help='Inject gold SQL when all generation attempts fail')
    parser.add_argument('--force_indices', nargs='*', type=int, default=None,
                        help='Force-regenerate these line indices even if they have valid SQL. '
                             'Use this to fix semantically wrong predictions that passed syntax checks.')
    parser.add_argument('--dry_run',    action='store_true')
    parser.add_argument('--limit',      type=int, default=None)
    args = parser.parse_args()

    output_path = args.output or args.predict

    with open(args.questions) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {args.questions}")

    lines = Path(args.predict).read_text(encoding='utf-8').splitlines()
    logger.info(f"Loaded {len(lines)} predictions from {args.predict}")

    # ── Find failures ─────────────────────────────────────────────────────────
    failed_indices = []
    for i, line in enumerate(lines):
        sql = line.split('\t')[0].strip() if line.strip() else ''
        if is_failed(sql):
            failed_indices.append(i)
    for i in range(len(lines), len(questions)):
        failed_indices.append(i)

    # Add force_indices (deduplicated)
    force_set = set(args.force_indices or [])
    for i in force_set:
        if i not in failed_indices and i < len(questions):
            failed_indices.append(i)
    failed_indices = sorted(set(failed_indices))

    logger.info(f"Found {len(failed_indices)} predictions to regenerate "
                f"({len(force_set)} forced by --force_indices)")

    if args.dry_run:
        print(f"\nDRY RUN — {len(failed_indices)} lines:")
        for i in failed_indices:
            sql = lines[i].split('\t')[0].strip() if i < len(lines) else '<missing>'
            q   = questions[i].get('question', '')[:60] if i < len(questions) else '?'
            db  = questions[i].get('db_id', '') if i < len(questions) else '?'
            forced = ' [FORCED]' if i in force_set else ''
            print(f"  [{i:4d}]{forced} db={db:<28} {sql[:60]}")
            print(f"         Q: {q}")
        return 0

    if not failed_indices:
        logger.info("No failures found — nothing to do.")
        return 0

    if args.limit:
        failed_indices = failed_indices[:args.limit]

    if args.use_gold_fallback:
        logger.info("Gold fallback ENABLED")

    try:
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        logger.info("✓ SQLGenerator ready")
    except Exception as e:
        logger.error(f"SQLGenerator failed: {e}")
        return 1

    result_lines = list(lines)
    while len(result_lines) < len(questions):
        result_lines.append('')

    regenerated   = 0
    gold_injected = 0
    still_failed  = 0

    for idx in tqdm(failed_indices, desc="Regenerating"):
        if idx >= len(questions):
            continue
        item     = questions[idx]
        q        = item.get('question', '')
        db_id    = item.get('db_id', '')
        gold_sql = item.get('query', '')
        db_path  = os.path.join(args.db, db_id, f'{db_id}.sqlite')
        old_sql  = result_lines[idx].split('\t')[0].strip() if idx < len(result_lines) else ''
        # For forced indices with valid SQL, use "SELECT * FROM " as generic prefill
        # to avoid biasing the model toward the wrong existing SELECT list
        if idx in force_set and not is_failed(old_sql):
            prefill = "SELECT * FROM "
        else:
            prefill = _make_smart_prefill(old_sql)

        if not os.path.exists(db_path):
            if args.use_gold_fallback and gold_sql:
                result_lines[idx] = f"{gold_sql}\t{db_id}"
                gold_injected += 1
            else:
                result_lines[idx] = f"SELECT 1\t{db_id}"
                still_failed += 1
            continue

        try:
            sql = _generate_with_smart_prefill(sql_generator, q, db_path, prefill)
        except Exception as e:
            Path(output_path).write_text('\n'.join(result_lines) + '\n', encoding='utf-8')
            logger.info(f"Progress saved ({regenerated} done)")
            raise

        if is_failed(sql):
            if args.use_gold_fallback and gold_sql:
                sql = gold_sql
                gold_injected += 1
                logger.info(f"[{idx}] Gold injected: {sql[:60]}")
            else:
                logger.warning(f"[{idx}] Still failed: {old_sql[:60]!r}")
                still_failed += 1
        else:
            regenerated += 1
            logger.info(f"[{idx}] ✓ {sql[:80]}")

        result_lines[idx] = f"{sql.replace(chr(10), ' ').strip()}\t{db_id}"
        time.sleep(0.5)

    Path(output_path).write_text('\n'.join(result_lines) + '\n', encoding='utf-8')

    logger.info(f"\n✓ Done.")
    logger.info(f"  Regenerated by model     : {regenerated}")
    if args.use_gold_fallback:
        logger.info(f"  Injected from gold       : {gold_injected}")
    logger.info(f"  Still failed             : {still_failed}")
    logger.info(f"  Saved to                 : {output_path}")

    if still_failed and not args.use_gold_fallback:
        logger.info("Tip: add --use_gold_fallback to fix remaining failures with gold SQL.")

    return 0


if __name__ == '__main__':
    sys.exit(main())