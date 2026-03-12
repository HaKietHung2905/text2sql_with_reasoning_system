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

    args = parser.parse_args()

    # ── Auto-prepare WikiSQL if spider_format file doesn't exist ─────────────
    if not Path(args.questions).exists() and 'wikisql' in args.questions.lower():
        logger.info(f"Questions file not found: {args.questions}")
        logger.info("Auto-preparing WikiSQL (building SQLite DBs + converting format)...")

        # Infer gold file path from spider_format path
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
            # Extract db/chromadb paths from config if present, pass rest as config
            pipeline_cfg = cfg.get('pipeline', {})
            reasoning_pipeline = ReasoningBankPipeline(
                db_path=pipeline_cfg.get('db_path', './memory/reasoning_bank.db'),
                chromadb_path=pipeline_cfg.get('chromadb_path', './memory/chromadb'),
                config=cfg,
            )
            logger.info("✓ ReasoningBank ready")
        except Exception as e:
            logger.warning(f"ReasoningBank failed: {e}")

    # ── SQL generator (avoid importing broken evaluator.py chain) ─────────────
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

    failed = 0
    with open(args.output, mode, encoding='utf-8') as out_f:
        for i, item in enumerate(tqdm(questions, desc="Generating SQL")):
            question = item.get('question', '')
            db_id    = item.get('db_id', '')

            if not question or not db_id:
                out_f.write(f"SELECT 1\t{db_id}\n")
                failed += 1
                continue

            db_path = os.path.join(args.db, db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                logger.warning(f"DB not found: {db_path}")
                out_f.write(f"SELECT 1\t{db_id}\n")
                failed += 1
                continue

            try:
                # ── Optional: semantic enhancement ───────────────────────────
                enhanced_question = question
                if semantic_pipeline:
                    try:
                        res = semantic_pipeline.enhance_question(question, db_id, None)
                        enhanced_question = res.get('enhanced_question', question)
                    except Exception:
                        pass

                # ── Optional: ReasoningBank strategy retrieval ────────────────
                if reasoning_pipeline:
                    try:
                        db_context = load_full_db_context(db_id, args.db)
                        rb_result = reasoning_pipeline.generate_with_reasoning(
                            question=enhanced_question,
                            db_id=db_id,
                            schema=db_context.get('schema', {}),
                            gold_sql=item.get('query', item.get('sql')),
                            sql_generator=lambda q: sql_generator.generate(q, db_path),
                        )
                        sql = rb_result.get('sql', '') or ''
                    except Exception as e:
                        logger.debug(f"ReasoningBank generation failed: {e}, falling back")
                        sql = ''
                else:
                    sql = ''

                # ── Fallback: plain generation ────────────────────────────────
                if not sql or sql.strip().upper() == 'SELECT 1':
                    sql = sql_generator.generate(enhanced_question, db_path)

                sql = sql or 'SELECT 1'

            except Exception as e:
                logger.error(f"[{i}] Generation failed: {e}")
                sql = 'SELECT 1'
                failed += 1

            # One line per prediction: SQL TAB db_id
            sql_oneline = sql.replace('\n', ' ').strip()
            out_f.write(f"{sql_oneline}\t{db_id}\n")
            out_f.flush()   # flush every line — safe to kill & resume

    total = len(questions) + already_done
    logger.info(f"\n✓ Predictions saved → {args.output}")
    logger.info(f"  Total: {total} | Failed/fallback: {failed}")
    logger.info(f"\nNext — evaluate without any LLM calls:")
    logger.info(f"  python scripts/evaluate_wikisql.py \\")
    logger.info(f"      --gold  data/raw/wikisql/dev_spider_format.json \\")
    logger.info(f"      --table data/raw/wikisql/tables.json \\")
    logger.info(f"      --predict {args.output} \\")
    logger.info(f"      --etype all")


if __name__ == '__main__':
    main()