"""
Main Spider evaluation script.
Entry point for all evaluation tasks.
"""
import warnings
import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import evaluate
from src.evaluation.foreign_key_mapper import build_foreign_key_map_from_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file"""
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

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Text-to-SQL system with ReasoningBank',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # === REQUIRED ARGUMENTS ===
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        help='Path to gold queries file (TSV or JSON)'
    )
    
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='Path to database directory'
    )
    
    # === OPTIONAL: PREDICTIONS ===
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help='Path to predicted queries file (TSV)'
    )
    
    parser.add_argument(
        '--questions',
        type=str,
        default=None,
        help='Path to questions file for generation'
    )
    
    # === EVALUATION TYPE ===
    parser.add_argument(
        '--etype',
        type=str,
        default='all',
        choices=['all', 'exec', 'match'],
        help='Evaluation type: all, exec (execution only), or match (exact match only)'
    )
    
    # === GENERATION OPTIONS ===
    parser.add_argument(
        '--use_langchain',
        action='store_true',
        help='Generate SQL using LangChain instead of loading predictions'
    )
    
    parser.add_argument(
        '--prompt_type',
        type=str,
        default='enhanced',
        choices=['basic', 'few_shot', 'chain_of_thought', 'enhanced'],
        help='Prompt template type'
    )
    
    parser.add_argument(
        '--enable_debugging',
        action='store_true',
        help='Enable SQL debugging and correction'
    )
    
    # === CHROMADB OPTIONS ===
    parser.add_argument(
        '--use_chromadb',
        action='store_true',
        help='Use ChromaDB for retrieval-augmented generation'
    )
    
    parser.add_argument(
        '--chromadb_config',
        type=str,
        default=None,
        help='Path to ChromaDB configuration file'
    )
    
    # === SEMANTIC LAYER OPTIONS ===
    parser.add_argument(
        '--use_semantic',
        action='store_true',
        help='Use semantic layer for query enhancement'
    )
    
    parser.add_argument(
        '--semantic_config',
        type=str,
        default=None,
        help='Path to semantic layer configuration file'
    )
    
    # === REASONINGBANK OPTIONS ===
    parser.add_argument(
        '--use_reasoning_bank',
        action='store_true',
        help='Use ReasoningBank for self-evolving memory and learning'
    )
    
    parser.add_argument(
        '--reasoning_config',
        type=str,
        default='./configs/reasoning_config.yaml',
        help='Path to ReasoningBank configuration file'
    )
    
    parser.add_argument(
        '--enable_test_time_scaling',
        action='store_true',
        help='Enable test-time scaling for difficult queries'
    )
    
    parser.add_argument(
        '--consolidation_frequency',
        type=int,
        default=50,
        help='How often to consolidate memory (every N queries)'
    )
    
    # === OTHER OPTIONS ===
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of examples to evaluate'
    )
    
    parser.add_argument(
        '--plug_value',
        action='store_true',
        help='Use gold values for execution evaluation'
    )
    
    parser.add_argument(
        '--keep_distinct',
        action='store_true',
        help='Keep DISTINCT in queries'
    )
    
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show progress bar for each datapoint'
    )
    
    parser.add_argument(
        '--tables',
        type=str,
        default=None,
        help='Path to tables.json file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results/evaluation_results.json',
        help='Path to save detailed results'
    )
    
    args = parser.parse_args()
    
    # === VALIDATE ARGUMENTS ===
    if args.use_langchain and not args.questions:
        parser.error("--use_langchain requires --questions")
    
    if not args.use_langchain and not args.predict:
        parser.error("Either --use_langchain or --predict required")
    
    # === LOAD CONFIGURATIONS ===
    chromadb_config = None
    if args.chromadb_config:
        chromadb_config = load_config(args.chromadb_config)
    
    semantic_config = None
    if args.semantic_config:
        semantic_config = load_config(args.semantic_config)
    elif args.use_semantic:
        # Use default semantic config
        semantic_config = {'enabled': True}
    
    reasoning_config = None
    if args.use_reasoning_bank:
        reasoning_config = load_config(args.reasoning_config)
        
        # Override with CLI arguments
        if not reasoning_config:
            reasoning_config = {}
        
        reasoning_config['enable_test_time_scaling'] = args.enable_test_time_scaling
        reasoning_config['consolidation_frequency'] = args.consolidation_frequency
    
    # === LOAD KMAPS ===
    kmaps = None
    if args.tables:
        with open(args.tables, 'r') as f:
            tables_data = json.load(f)
            kmaps = {table['db_id']: table for table in tables_data}
    else:
        # Create empty kmaps
        kmaps = {}
    
    # === PRINT CONFIGURATION ===
    logger.info("="*80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Gold file: {args.gold}")
    logger.info(f"Database directory: {args.db}")
    logger.info(f"Evaluation type: {args.etype}")
    logger.info(f"Generation mode: {'LangChain' if args.use_langchain else 'Predictions file'}")
    logger.info(f"Prompt type: {args.prompt_type}")
    logger.info(f"ChromaDB enabled: {args.use_chromadb}")
    logger.info(f"Semantic layer enabled: {args.use_semantic}")
    logger.info(f"ReasoningBank enabled: {args.use_reasoning_bank}")
    if args.use_reasoning_bank:
        logger.info(f"  - Test-time scaling: {args.enable_test_time_scaling}")
        logger.info(f"  - Consolidation frequency: {args.consolidation_frequency}")
    logger.info(f"Limit: {args.limit if args.limit else 'None'}")
    logger.info("="*80)
    
    # === RUN EVALUATION ===
    try:
        results = evaluate(
            gold=args.gold,
            predict=args.predict,
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
            limit=args.limit
        )
        
        # === SAVE RESULTS ===
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
        
        # === PRINT SUMMARY ===
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Exact Match Accuracy: {results['exact_match_accuracy']:.2%}")
        logger.info(f"Execution Accuracy: {results['execution_accuracy']:.2%}")
        
        if 'semantic_statistics' in results:
            logger.info("\nSemantic Layer Statistics:")
            for key, value in results['semantic_statistics'].items():
                logger.info(f"  {key}: {value}")
        
        if 'reasoning_statistics' in results:
            logger.info("\nReasoningBank Statistics:")
            for key, value in results['reasoning_statistics'].items():
                logger.info(f"  {key}: {value}")
        
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for Spider exact match comparison.
    Handles newlines, whitespace, and formatting differences.
    """
    if not sql or not sql.strip():
        return sql
    
    # Remove newlines and collapse whitespace
    sql = re.sub(r'\s+', ' ', sql)
    
    # Lowercase keywords
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 
        'ORDER', 'LIMIT', 'JOIN', 'ON', 'AND', 'OR', 'IN',
        'NOT', 'LIKE', 'BETWEEN', 'DISTINCT', 'AS', 'UNION',
        'INTERSECT', 'EXCEPT'
    ]
    for kw in keywords:
        sql = re.sub(rf'\b{kw}\b', kw.lower(), sql, flags=re.IGNORECASE)
    
    # Uppercase aggregate functions (Spider convention)
    functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    for fn in functions:
        sql = re.sub(rf'\b{fn}\b', fn, sql, flags=re.IGNORECASE)
    
    # Normalize punctuation spacing
    sql = re.sub(r'\s*,\s*', ' , ', sql)
    sql = re.sub(r'\s*\(\s*', ' ( ', sql)
    sql = re.sub(r'\s*\)\s*', ' ) ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    
    # Final cleanup
    sql = re.sub(r'\s+', ' ', sql)
    
    return sql.strip()

def print_configuration(args):
    """Print evaluation configuration"""
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Gold queries:     {args.gold}")
    print(f"Predictions:      {args.pred if args.pred else 'Generated from questions'}")
    print(f"Database dir:     {args.db}")
    print(f"Tables file:      {args.table}")
    print(f"Evaluation type:  {args.etype}")
    
    if args.limit:
        print(f"Limit:            {args.limit} examples")
    
    if args.use_langchain:
        print(f"\nSQL Generation:")
        print(f"  Questions file: {args.questions}")
        print(f"  Prompt type:    {args.prompt_type}")
        print(f"  Debugging:      {args.enable_debugging}")
    
    if args.use_chromadb:
        print(f"\nChromaDB:")
        print(f"  Data dir:       {args.chromadb_data_dir}")
        print(f"  Persist dir:    {args.chromadb_persist_dir}")
        print(f"  N examples:     {args.chromadb_n_examples}")
    
    if args.use_semantic:
        print(f"\nSemantic Layer:   ✓ Enabled")
    
    print("="*70 + "\n")


def save_results(results: dict, output_path: str):
    """Save results to file"""
    import json
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())