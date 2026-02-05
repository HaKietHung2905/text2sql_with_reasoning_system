"""
Main Spider evaluation script.
Entry point for all evaluation tasks.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import evaluate
from src.evaluation.foreign_key_mapper import build_foreign_key_map_from_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate Text-to-SQL predictions on Spider dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic exact match evaluation
  python scripts/evaluate_spider.py \\
      --gold data/raw/spider/dev.json \\
      --pred results/predictions.txt \\
      --db data/raw/spider/database \\
      --table data/raw/spider/tables.json \\
      --etype match

  # Full evaluation (execution + exact match)
  python scripts/evaluate_spider.py \\
      --gold data/raw/spider/dev.json \\
      --pred results/predictions.txt \\
      --db data/raw/spider/database \\
      --table data/raw/spider/tables.json \\
      --etype all

  # Generate SQL from questions using LangChain
  python scripts/evaluate_spider.py \\
      --gold data/raw/spider/dev.json \\
      --db data/raw/spider/database \\
      --table data/raw/spider/tables.json \\
      --use_langchain \\
      --questions questions/questions.txt \\
      --etype all

  # With ChromaDB retrieval
  python scripts/evaluate_spider.py \\
      --gold data/raw/spider/dev.json \\
      --db data/raw/spider/database \\
      --table data/raw/spider/tables.json \\
      --use_langchain \\
      --questions questions/questions.txt \\
      --use_chromadb \\
      --chromadb_data_dir ./data/raw/spider \\
      --etype all

  # With semantic layer
  python scripts/evaluate_spider.py \\
      --gold data/raw/spider/dev.json \\
      --db data/raw/spider/database \\
      --table data/raw/spider/tables.json \\
      --use_langchain \\
      --questions questions/questions.txt \\
      --use_semantic \\
      --etype all
        """
    )
    
    # Required arguments
    parser.add_argument('--gold', required=True, type=str,
                       help='Path to gold queries file')
    parser.add_argument('--db', required=True, type=str,
                       help='Directory containing database files')
    parser.add_argument('--table', required=True, type=str,
                       help='Path to tables.json schema file')
    
    # Optional prediction file
    parser.add_argument('--pred', type=str,
                       help='Path to predicted queries file')
    
    # Evaluation type
    parser.add_argument('--etype', type=str, default='all',
                       choices=['all', 'exec', 'match'],
                       help='Evaluation type: all, exec, or match')
    
    # Evaluation options
    parser.add_argument('--plug_value', action='store_true',
                       help='Plug gold values into predicted queries')
    parser.add_argument('--keep_distinct', action='store_true',
                       help='Keep DISTINCT keyword during evaluation')
    parser.add_argument('--progress_bar_for_each_datapoint', action='store_true',
                       help='Show progress bar for each datapoint')
    
    # SQL generation options
    parser.add_argument('--use_langchain', action='store_true',
                       help='Generate SQL from questions using LangChain')
    parser.add_argument('--questions', type=str,
                       help='Path to questions file (required with --use_langchain)')
    parser.add_argument('--prompt_type', type=str, default='enhanced',
                       choices=['basic', 'few_shot', 'chain_of_thought', 
                               'rule_based', 'enhanced', 'step_by_step'],
                       help='Prompting strategy type')
    
    # Enhancement options
    parser.add_argument('--enable_debugging', action='store_true',
                       help='Enable debugging prompts for SQL correction')
    parser.add_argument('--use_chromadb', action='store_true',
                       help='Enable ChromaDB retrieval-augmented generation')
    parser.add_argument('--use_semantic', action='store_true',
                       help='Enable semantic layer for SQL enhancement')
    
    # ChromaDB configuration
    parser.add_argument('--chromadb_data_dir', type=str, default='./data/raw/spider',
                       help='Spider dataset directory for ChromaDB')
    parser.add_argument('--chromadb_persist_dir', type=str, default='./data/embeddings/chroma_db',
                       help='ChromaDB persistence directory')
    parser.add_argument('--chromadb_n_examples', type=int, default=3,
                       help='Number of similar examples to retrieve')
    parser.add_argument('--chromadb_min_similarity', type=float, default=0.3,
                       help='Minimum similarity threshold for retrieval')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for detailed results (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_langchain and not args.questions:
        parser.error("--questions is required when using --use_langchain")
    
    if not args.use_langchain and not args.pred:
        parser.error("--pred is required when not using --use_langchain")
    
    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    print_configuration(args)
    
    # Build foreign key maps
    kmaps = None
    if args.etype in ['all', 'match']:
        logger.info("Building foreign key maps...")
        kmaps = build_foreign_key_map_from_json(args.table)
        logger.info(f"Built maps for {len(kmaps)} databases")
    
    # Prepare ChromaDB configuration
    chromadb_config = None
    if args.use_chromadb:
        chromadb_config = {
            'data_dir': args.chromadb_data_dir,
            'persist_dir': args.chromadb_persist_dir,
            'n_examples': args.chromadb_n_examples,
            'min_similarity': args.chromadb_min_similarity
        }
        logger.info(f"ChromaDB config: {chromadb_config}")
    
    # Run evaluation
    logger.info("\n" + "="*70)
    logger.info("Starting Spider Evaluation")
    logger.info("="*70 + "\n")
    
    try:
        results = evaluate(
            gold=args.gold,
            predict=args.pred,
            db_dir=args.db,
            etype=args.etype,
            kmaps=kmaps,
            plug_value=args.plug_value,
            keep_distinct=args.keep_distinct,
            progress_bar_for_each_datapoint=args.progress_bar_for_each_datapoint,
            use_langchain=args.use_langchain,
            questions_file=args.questions,
            prompt_type=args.prompt_type,
            enable_debugging=args.enable_debugging,
            use_chromadb=args.use_chromadb,
            chromadb_config=chromadb_config,
            use_semantic=args.use_semantic
        )
        
        # Print final results
        logger.info("\n" + "="*70)
        logger.info("Evaluation Complete!")
        logger.info("="*70)
        logger.info(f"Results: {results}")
        
        # Save results if output specified
        if args.output:
            save_results(results, args.output)
            logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


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
        print(f"\nSemantic Layer:   Enabled")
    
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