"""
Script to convert Spider dataset JSON files to various text formats.

This script converts Spider JSON format to tab-separated text files
suitable for evaluation and training.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spider_converter import SpiderConverter, convert_spider_dataset
from utils.logging_utils import get_logger
from utils.config_loader import load_config

logger = get_logger(__name__)


def main():
    """Main conversion script"""
    parser = argparse.ArgumentParser(
        description='Convert Spider JSON dataset to text formats'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input Spider JSON file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for converted files'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='dev',
        help='Dataset name (e.g., dev, train, test)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['sql', 'questions', 'full', 'all'],
        default='all',
        help='Output format to generate'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting Spider dataset: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    
    converter = SpiderConverter()
    
    if args.format == 'all':
        # Convert to all formats
        output_files = convert_spider_dataset(
            args.input,
            args.output_dir,
            args.dataset_name
        )
        
        logger.info("Conversion complete! Generated files:")
        for format_type, filepath in output_files.items():
            logger.info(f"  {format_type}: {filepath}")
    
    else:
        # Convert to specific format
        output_file = f"{args.output_dir}/{args.dataset_name}_{args.format}.txt"
        
        if args.format == 'sql':
            converter.json_to_sql_txt(args.input, output_file)
        elif args.format == 'questions':
            converter.json_to_questions_txt(args.input, output_file)
        elif args.format == 'full':
            converter.json_to_full_format(args.input, output_file)
        
        logger.info(f"Conversion complete! Generated: {output_file}")


if __name__ == "__main__":
    main()