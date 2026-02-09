"""
Script to download and setup WikiSQL dataset.

This script downloads the WikiSQL dataset from HuggingFace
and prepares it for use in the evaluation system.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.wikisql_downloader import WikiSQLDatasetDownloader
from src.data.wikisql_loader import WikiSQLDataLoader, WikiSQLSchemaExtractor
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main download and setup script"""
    parser = argparse.ArgumentParser(
        description='Download and setup WikiSQL dataset'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/raw',
        help='Directory to store dataset'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing dataset'
    )
    
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Explore dataset after download'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of sample examples to show'
    )
    
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export dev split to CSV'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = WikiSQLDatasetDownloader(args.data_dir)
    
    if args.verify:
        # Just verify existing dataset
        logger.info("🔍 Verifying dataset...")
        stats = downloader.verify_dataset()
        
        if stats:
            logger.info("Dataset verification:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.error("Dataset not found")
        
        return
    
    # Download dataset
    logger.info("📊 WikiSQL Dataset Downloader")
    logger.info("=" * 50)
    
    logger.info("Step 1: Downloading WikiSQL dataset...")
    
    success = downloader.download_dataset()
    
    if not success:
        logger.error("❌ Download failed")
        return
    
    # Verify download
    logger.info("\nStep 2: Verifying download...")
    stats = downloader.verify_dataset()
    
    if stats:
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    if not args.explore and not args.export_csv:
        logger.info("\n✅ Download complete!")
        logger.info("💡 Use --explore to see sample data")
        logger.info("💡 Use --export-csv to export to CSV format")
        return
    
    # Explore dataset
    if args.explore:
        logger.info("\nStep 3: Exploring dataset...")
        
        loader = WikiSQLDataLoader(f"{args.data_dir}/wikisql")
        schema_extractor = WikiSQLSchemaExtractor()
        
        # Show overall stats
        dataset_stats = loader.get_stats()
        logger.info("\n=== Dataset Statistics ===")
        for key, value in dataset_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Show samples
        logger.info(f"\n=== Sample Training Examples (n={args.samples}) ===")
        samples = loader.get_samples("train", args.samples)
        
        for i, sample in enumerate(samples, 1):
            logger.info(f"\nExample {i}:")
            logger.info(f"  Question: {sample['question']}")
            logger.info(f"  SQL: {sample.get('sql', 'N/A')}")
            logger.info(f"  Table: {sample.get('table', {}).get('name', 'N/A')}")
            
            # Show schema
            schema_str = schema_extractor.format_schema_string(sample)
            logger.info(f"  Schema: {schema_str.replace(chr(10), ' | ')}")
            logger.info("-" * 50)
        
        # Show difficulty distribution
        logger.info("\n=== Difficulty Distribution (Dev Set) ===")
        difficulty_groups = loader.filter_by_difficulty("dev")
    
    # Export to CSV
    if args.export_csv:
        logger.info("\nStep 4: Exporting to CSV...")
        loader = WikiSQLDataLoader(f"{args.data_dir}/wikisql")
        
        for split in ['train', 'dev', 'test']:
            output_path = Path(args.data_dir) / "wikisql" / f"{split}_export.csv"
            loader.export_to_csv(split, str(output_path))
    
    logger.info("\n✅ Setup complete!")


if __name__ == "__main__":
    main()