"""
Step 1: Generate SQL predictions and save to a TSV file.
Step 2: Run evaluate_wikisql.py / evaluate_spider.py with --predict <file>
        for fast, LLM-free evaluation.

Usage (WikiSQL):
  python scripts/generate_predictions.py \
      --questions data/raw/wikisql/dev_spider_format.json \
      --db        data/raw/wikisql/database \
      --output    results/predictions_wikisql.tsv \
      --use_reasoning_bank --use_chromadb --use_semantic \
      --limit 50

Usage (Spider):
  python scripts/generate_predictions.py \
      --questions data/spider/dev.json \
      --db        data/spider/database \
      --output    results/predictions_spider.tsv
"""

import sys
import os
import json
import argparse
import logging
import warnings
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

# ── Logging ───────────────────────────────────────────────────────────────────
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)
for _n in ['chromadb', 'chromadb.api', 'chromadb.telemetry',
           'utils.embedding_utils', 'src.reasoning.memory_retrieval',
           'src.reasoning.memory_store', 'src.reasoning.reasoning_pipeline']:
    logging.getLogger(_n).setLevel(logging.ERROR)

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
logger = get_logger(__name__)


def load_config(path):
    if not path or not os.path.exists(path):
        return {}
    if path.endswith('.json'):
        return json.load(open(path))
    try:
        import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        return {}


def _is_server_error(exc: Exception) -> bool:
    """
    Return True if exc (or any chained cause) is an HTTP 5xx server error.
    Walks __cause__ and __context__ so wrapped exceptions are also detected.
    """
    _5xx = ("500", "502", "503", "504",
            "Internal Server Error", "Bad Gateway",
            "Service Unavailable", "Gateway Timeout")

    seen = set()
    node = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if any(code in str(node) for code in _5xx):
            return True
        if any(code in repr(node) for code in _5xx):
            return True
        node = node.__cause__ or node.__context__

    return False


def _stop_on_server_error(exc: Exception, out_f, already_done: int, i: int,
                           output_path: str) -> None:
    """
    On a 5xx server error: flush the checkpoint, wait, then re-exec this
    process with --resume so generation continues automatically.
    """
    if not _is_server_error(exc):
        return
    out_f.flush()
    done_so_far = already_done + i
    wait_seconds = 60

    logger.error(
        f"\n{'='*60}\n"
        f"❌  Server error (5xx) at question {done_so_far}.\n"
        f"    Checkpoint saved: {output_path} ({done_so_far} lines)\n"
        f"    Waiting {wait_seconds}s then auto-resuming...\n"
        f"{'='*60}"
    )

    time.sleep(wait_seconds)

    new_argv = sys.argv[:]
    if '--resume' not in new_argv:
        new_argv.append('--resume')

    logger.info(f"🔄 Auto-resuming: {' '.join(new_argv)}")
    os.execv(sys.executable, [sys.executable] + new_argv)
    # os.execv replaces the current process — nothing after this runs


def main():
    parser = argparse.ArgumentParser(description='Generate SQL predictions to TSV file')

    parser.add_argument('--questions', required=True,
                        help='JSON questions file (Spider or WikiSQL spider-format)')
    parser.add_argument('--db', required=True,
                        help='Database directory')
    parser.add_argument('--output', default='./results/predictions.tsv',
                        help='Output TSV file (sql TAB db_id per line)')

    # Feature flags
    parser.add_argument('--use_chromadb',      action='store_true')
    parser.add_argument('--chromadb_config',   default=None)
    parser.add_argument('--use_semantic',      action='store_true')
    parser.add_argument('--semantic_config',   default=None)
    parser.add_argument('--use_reasoning_bank', action='store_true')
    parser.add_argument('--reasoning_config',
                        default='./configs/reasoning_config.yaml')

    parser.add_argument('--limit',  type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-generated lines (resume interrupted run)')
    parser.add_argument('--checkpoint_size', type=int, default=None,
                        help='Stop after generating this many NEW predictions '
                             '(re-run with --resume to continue)')

    args = parser.parse_args()

    # ── Auto-prepare WikiSQL if spider_format file doesn't exist ─────────────
    if not Path(args.questions).exists() and 'wikisql' in args.questions.lower():
        logger.info(f"Questions file not found: {args.questions}")
        logger.info("Auto-preparing WikiSQL (building SQLite DBs + converting format)...")
        gold_file = args.questions.replace('_spider_format.json', '.json')
        if not Path(gold_file).exists():
            logger.error(f"Cannot find WikiSQL gold file at: {gold_file}")
            sys.exit(1)
        from scripts.evaluate_wikisql import (
            prepare_wikisql_databases,
            convert_wikisql_gold_to_spider_format,
        )
        prepare_wikisql_databases(gold_file=gold_file, db_dir=args.db, limit=args.limit)
        convert_wikisql_gold_to_spider_format(
            gold_file=gold_file, output_file=args.questions, limit=args.limit
        )
        logger.info(f"✓ Spider-format file created: {args.questions}")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(args.questions, 'r') as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]
    logger.info(f"Loaded {len(questions)} questions")

    # ── Resume support ────────────────────────────────────────────────────────
    already_done = 0
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            already_done = sum(1 for ln in f if ln.strip())
        logger.info(f"Resuming from question {already_done}")
        questions = questions[already_done:]

    # ── Init optional pipelines ───────────────────────────────────────────────
    semantic_pipeline  = None
    reasoning_pipeline = None

    if args.use_semantic:
        try:
            from src.semantic.semantic_pipeline import SemanticPipeline
            cfg = load_config(args.semantic_config) or {'enabled': True}
            semantic_pipeline = SemanticPipeline(cfg)
            logger.info("✓ Semantic pipeline ready")
        except Exception as e:
            logger.warning(f"Semantic pipeline failed: {e}")

    if args.use_reasoning_bank:
        try:
            from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
            cfg = load_config(args.reasoning_config) or {}
            pipeline_cfg = cfg.get('pipeline', {})
            reasoning_pipeline = ReasoningBankPipeline(
                db_path=pipeline_cfg.get('db_path', './memory/reasoning_bank.db'),
                chromadb_path=pipeline_cfg.get('chromadb_path', './memory/chromadb'),
                config=cfg,
            )
            logger.info("✓ ReasoningBank ready")
        except Exception as e:
            logger.warning(f"ReasoningBank failed: {e}")

    # ── SQL generator ─────────────────────────────────────────────────────────
    try:
        from src.generation.sql_generator import SQLGenerator
        sql_generator = SQLGenerator()
        logger.info("✓ SQLGenerator ready")
    except Exception as e:
        logger.error(f"SQLGenerator failed to load: {e}")
        sys.exit(1)

    # ── Schema loader ─────────────────────────────────────────────────────────
    from utils.sql_schema import load_full_db_context

    # ── Generate ──────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if args.resume else 'w'

    failed   = 0
    new_count = 0

    with open(args.output, mode, encoding='utf-8') as out_f:
        for i, item in enumerate(tqdm(questions, desc="Generating SQL")):
            question = item.get('question', '')
            db_id    = item.get('db_id', '')

            if not question or not db_id:
                out_f.write(f"SELECT 1\t{db_id}\n")
                out_f.flush()
                failed    += 1
                new_count += 1

            else:
                db_path = os.path.join(args.db, db_id, f"{db_id}.sqlite")
                if not os.path.exists(db_path):
                    logger.warning(f"DB not found: {db_path}")
                    out_f.write(f"SELECT 1\t{db_id}\n")
                    out_f.flush()
                    failed    += 1
                    new_count += 1

                else:
                    try:
                        # ── Semantic enhancement ──────────────────────────────
                        enhanced_question = question
                        if semantic_pipeline:
                            try:
                                res = semantic_pipeline.enhance_question(
                                    question, db_id, None)
                                enhanced_question = res.get('enhanced_question', question)
                            except Exception:
                                pass

                        # ── ReasoningBank ─────────────────────────────────────
                        sql = ''
                        if reasoning_pipeline:
                            try:
                                db_context = load_full_db_context(db_id, args.db)
                                rb_result = reasoning_pipeline.generate_with_reasoning(
                                    question=enhanced_question,
                                    db_id=db_id,
                                    schema=db_context.get('schema', {}),
                                    gold_sql=item.get('query', item.get('sql')),
                                    sql_generator=lambda q: sql_generator.generate(
                                        q, db_path),
                                )
                                sql = rb_result.get('sql', '') or ''
                            except Exception as e:
                                # Re-raise 5xx immediately — do NOT fall back
                                _stop_on_server_error(e, out_f, already_done, i,
                                                      args.output)
                                logger.debug(f"ReasoningBank failed: {e}, falling back")
                                sql = ''

                        # ── Plain generation fallback ─────────────────────────
                        if not sql or sql.strip().upper() == 'SELECT 1':
                            sql = sql_generator.generate(enhanced_question, db_path)

                        sql = sql or 'SELECT 1'

                    except Exception as e:
                        # DEBUG — remove after confirming fix
                        _cause_str = repr(str(e.__cause__)[:200]) if e.__cause__ else None
                        _cause_type = type(e.__cause__).__name__ if e.__cause__ else None
                        print(f"[DEBUG] type={type(e).__name__}", flush=True)
                        print(f"[DEBUG] str={repr(str(e)[:200])}", flush=True)
                        print(f"[DEBUG] cause_type={_cause_type}", flush=True)
                        print(f"[DEBUG] cause_str={_cause_str}", flush=True)
                        print(f"[DEBUG] _is_server_error={_is_server_error(e)}", flush=True)
                        # Re-raise 5xx immediately — do NOT write SELECT 1
                        _stop_on_server_error(e, out_f, already_done, i, args.output)
                        logger.error(f"[{already_done + i}] Generation failed: {e}")
                        sql = 'SELECT 1'
                        failed += 1

                    sql_oneline = sql.replace('\n', ' ').strip()
                    out_f.write(f"{sql_oneline}\t{db_id}\n")
                    out_f.flush()
                    new_count += 1

            # ── Checkpoint size check ─────────────────────────────────────────
            if args.checkpoint_size and new_count >= args.checkpoint_size:
                logger.info(
                    f"\n✓ Checkpoint: {new_count} new predictions written "
                    f"({already_done + new_count} total). "
                    f"Re-run with --resume to continue."
                )
                break

    total = already_done + new_count
    logger.info(f"\n✓ Predictions saved → {args.output}")
    logger.info(f"  Total: {total} | New this run: {new_count} | Failed/fallback: {failed}")
    logger.info(f"\nNext — evaluate without any LLM calls:")
    logger.info(f"  python scripts/evaluate_wikisql.py \\")
    logger.info(f"      --gold  data/raw/wikisql/dev_spider_format.json \\")
    logger.info(f"      --table data/raw/wikisql/tables.json \\")
    logger.info(f"      --predict {args.output} \\")
    logger.info(f"      --etype all")


if __name__ == '__main__':
    main()