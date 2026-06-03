"""
Main Spider evaluation script.
Entry point for all evaluation tasks.
"""
import warnings
import sys
import os
import argparse
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
logging.getLogger('__main__').setLevel(logging.INFO)
logging.getLogger('src.reasoning.evaluator').setLevel(logging.INFO)

for logger_name in [
    'utils.embedding_utils',
    'src.reasoning.memory_retrieval',
    'src.reasoning.memory_store',
    'src.reasoning.reasoning_pipeline',
    'src.reasoning.experience_collector',
    'src.reasoning.self_judgment',
    'src.reasoning.strategy_distillation',
    'src.reasoning.memory_consolidation',
    'src.semantic.semantic_pipeline',
    'chromadb', 'chromadb.api', 'chromadb.telemetry',
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

warnings.filterwarnings('ignore')
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.evaluator import evaluate
from src.evaluation.foreign_key_mapper import build_foreign_key_map_from_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML config")
            return {}
    else:
        logger.warning(f"Unsupported config format: {config_path}")
        return {}


def save_summary_report(results: dict, output_json_path: str, dataset: str = "Spider") -> str:
    from datetime import datetime
    txt_path = Path(output_json_path).with_suffix(".txt")

    em  = results.get("exact_match_accuracy", 0)
    ex  = results.get("execution_accuracy",   0)
    n   = results.get("total_evaluated",      0)

    lines = [
        "=" * 80,
        f"{dataset.upper()} EVALUATION SUMMARY REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Results   : {output_json_path}",
        "=" * 80,
        "",
        "MAIN METRICS",
        "-" * 40,
        f"  Exact Match Accuracy : {em:.2%}  ({em*100:.1f}%)",
        f"  Execution Accuracy   : {ex:.2%}  ({ex*100:.1f}%)",
        f"  Total Evaluated      : {n}",
        "",
    ]

    if "semantic_statistics" in results:
        lines += ["SEMANTIC LAYER STATISTICS", "-" * 40]
        for k, v in results["semantic_statistics"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if "reasoning_statistics" in results:
        lines += ["REASONINGBANK STATISTICS", "-" * 40]
        for k, v in results["reasoning_statistics"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    scores = results.get("scores", {})
    if scores:
        lines += ["PER-DIFFICULTY BREAKDOWN", "-" * 40]
        for level in ["easy", "medium", "hard", "extra", "all"]:
            if level in scores:
                lvl   = scores[level]
                cnt   = lvl.get("count", 0)
                exact = lvl.get("exact", 0)
                exec_ = lvl.get("exec",  0)
                lines.append(f"  {level:<8} | count={cnt:<5} | EM={exact:.2%} | EX={exec_:.2%}")
        lines.append("")

    lines += ["=" * 80, "END OF REPORT", "=" * 80]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(txt_path)


def normalize_sql(sql: str) -> str:
    if not sql or not sql.strip():
        return sql
    sql = re.sub(r'\s+', ' ', sql)
    keywords = ['SELECT','FROM','WHERE','GROUP','BY','HAVING','ORDER','LIMIT',
                'JOIN','ON','AND','OR','IN','NOT','LIKE','BETWEEN','DISTINCT',
                'AS','UNION','INTERSECT','EXCEPT']
    for kw in keywords:
        sql = re.sub(rf'\b{kw}\b', kw.lower(), sql, flags=re.IGNORECASE)
    for fn in ['COUNT','SUM','AVG','MIN','MAX']:
        sql = re.sub(rf'\b{fn}\b', fn, sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s*,\s*', ' , ', sql)
    sql = re.sub(r'\s*\(\s*', ' ( ', sql)
    sql = re.sub(r'\s*\)\s*', ' ) ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Text-to-SQL system on Spider',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument('--gold',    type=str, required=True)
    parser.add_argument('--db',      type=str, required=True)

    # Predictions
    parser.add_argument('--predict',   type=str, default=None)
    parser.add_argument('--questions', type=str, default=None)

    # Evaluation type
    parser.add_argument('--etype', type=str, default='all',
                        choices=['all', 'exec', 'match'])

    # Generation options
    parser.add_argument('--use_langchain',    action='store_true')
    parser.add_argument('--prompt_type',      type=str, default='enhanced',
                        choices=['basic','few_shot','chain_of_thought','enhanced'])
    parser.add_argument('--enable_debugging', action='store_true')

    # ChromaDB
    parser.add_argument('--use_chromadb',    action='store_true')
    parser.add_argument('--chromadb_config', type=str, default=None)

    # Semantic layer
    parser.add_argument('--use_semantic',    action='store_true')
    parser.add_argument('--semantic_config', type=str, default=None)

    # ReasoningBank
    parser.add_argument('--use_reasoning_bank',       action='store_true')
    parser.add_argument('--reasoning_config',         type=str,
                        default='./configs/reasoning_config.yaml')
    parser.add_argument('--enable_test_time_scaling', action='store_true')
    parser.add_argument('--consolidation_frequency',  type=int, default=50)

    # Other
    parser.add_argument('--limit',         type=int,  default=None)
    parser.add_argument('--plug_value',    action='store_true')
    parser.add_argument('--keep_distinct', action='store_true')
    parser.add_argument('--progress',      action='store_true')
    parser.add_argument('--tables',        type=str, default=None)
    parser.add_argument('--output',        type=str,
                        default='./results/evaluation_results.json')

    args = parser.parse_args()

    if args.use_langchain and not args.questions:
        parser.error("--use_langchain requires --questions")
    if not args.use_langchain and not args.predict:
        parser.error("Either --use_langchain or --predict required")

    # Load configs
    chromadb_config  = load_config(args.chromadb_config) if args.chromadb_config else {}
    semantic_config  = (load_config(args.semantic_config) if args.semantic_config
                        else ({'enabled': True} if args.use_semantic else {}))
    reasoning_config = {}
    if args.use_reasoning_bank:
        reasoning_config = load_config(args.reasoning_config) or {}
        reasoning_config['enable_test_time_scaling'] = args.enable_test_time_scaling
        reasoning_config['consolidation_frequency']  = args.consolidation_frequency

    # Build foreign-key map
    kmaps = {}
    if args.tables and os.path.exists(args.tables):
        try:
            kmaps = build_foreign_key_map_from_json(args.tables)
            logger.info(f"✓ Foreign key map loaded from: {args.tables}")
        except Exception as e:
            logger.warning(f"Could not load foreign key map: {e}")

    # Print config
    logger.info("=" * 80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Gold file: {args.gold}")
    logger.info(f"Database directory: {args.db}")
    logger.info(f"Evaluation type: {args.etype}")
    logger.info(f"Generation mode: {'LangChain' if args.use_langchain else 'Predictions file'}")
    logger.info(f"Prompt type: {args.prompt_type}")
    logger.info(f"ChromaDB enabled: {args.use_chromadb}")
    logger.info(f"Semantic layer enabled: {args.use_semantic}")
    logger.info(f"ReasoningBank enabled: {args.use_reasoning_bank}")
    logger.info(f"Limit: {args.limit}")
    logger.info("=" * 80)

    try:
        # ── No preprocessing — model generates correct SQL with proper aliases ──
        # The old preprocessor (_strip_table_prefix) was destroying valid JOIN
        # ON clauses by stripping tN. from WHERE/ON conditions, converting
        # "ON t1.stadium_id = t2.stadium_id" into "ON stadium_id = stadium_id"
        # which causes SQLite "ambiguous column name" errors.
        predict_path = args.predict

        logger.info("\n" + "=" * 80)
        logger.info("TEXT-TO-SQL EVALUATION WITH REASONINGBANK")
        logger.info("=" * 80)

        results = evaluate(
            gold=args.gold,
            predict=predict_path,
            db_dir=args.db,
            etype=args.etype,
            kmaps=kmaps,
            plug_value=args.plug_value,
            keep_distinct=args.keep_distinct,
            progress_bar_for_each_datapoint=args.progress,
            use_langchain=args.use_langchain,
            questions_file=args.questions,
            prompt_type=args.prompt_type,
            enable_debugging=args.enable_debugging,
            use_chromadb=args.use_chromadb,
            chromadb_config=chromadb_config,
            use_semantic=args.use_semantic,
            semantic_config=semantic_config,
            use_reasoning_bank=args.use_reasoning_bank,
            reasoning_config=reasoning_config,
            limit=args.limit,
        )

        # Save results
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to: {output_file}")

        txt_file = save_summary_report(results, str(output_file), dataset="Spider")
        logger.info(f"✓ Summary report  : {txt_file}")

        # Terminal summary
        logger.info("\n" + "=" * 80)
        logger.info("SPIDER EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Exact Match Accuracy : {results.get('exact_match_accuracy', 0)*100:.2f}%")
        logger.info(f"Execution Accuracy   : {results.get('execution_accuracy',   0)*100:.2f}%")
        logger.info(f"Total Evaluated      : {results.get('total_evaluated', 0)}")

        scores = results.get("scores", {})
        if scores:
            logger.info("")
            logger.info("Per-difficulty breakdown:")
            for level in ["easy", "medium", "hard", "extra", "all"]:
                if level in scores:
                    lvl   = scores[level]
                    cnt   = lvl.get("count", 0)
                    exact = lvl.get("exact", 0)
                    exec_ = lvl.get("exec",  0)
                    logger.info(f"  {level:<8} | count={cnt:<5} | EM={exact:.2%} | EX={exec_:.2%}")

        if "semantic_statistics" in results:
            logger.info("\nSemantic Layer Statistics:")
            for k, v in results["semantic_statistics"].items():
                logger.info(f"  {k}: {v}")

        if "reasoning_statistics" in results:
            logger.info("\nReasoningBank Statistics:")
            for k, v in results["reasoning_statistics"].items():
                logger.info(f"  {k}: {v}")

        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())