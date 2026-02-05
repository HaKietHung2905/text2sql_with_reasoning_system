"""
Script to download and setup Spider dataset.

This script downloads the Spider dataset from either the full source
or HuggingFace, and prepares it for use.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spider_downloader import SpiderDatasetDownloader, SpiderDatabaseManager
from src.data.spider_loader import SpiderDataLoader
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main download and setup script"""
    parser = argparse.ArgumentParser(
        description='Download and setup Spider dataset'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/raw',
        help='Directory to store dataset'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Try to download full dataset with SQLite files'
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
        default=3,
        help='Number of sample examples to show'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = SpiderDatasetDownloader(args.data_dir)
    
    if args.verify:
        # Just verify existing dataset
        logger.info("🔍 Verifying dataset...")
        stats = downloader.verify_dataset()
        
        logger.info("Dataset verification:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return
    
    # Download dataset
    logger.info("🕷️ Spider Dataset Downloader")
    logger.info("=" * 50)
    
    logger.info("Step 1: Downloading Spider dataset...")
    
    if args.full:
        success = downloader.download_dataset(prefer_full=True)
    else:
        success = downloader.download_from_huggingface()
    
    if not success:
        logger.error("❌ Download failed")
        return
    
    # Verify download
    logger.info("\nStep 2: Verifying download...")
    stats = downloader.verify_dataset()
    
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    if not args.explore:
        logger.info("\n✅ Download complete!")
        logger.info(f"💡 Use --explore to see sample data")
        return
    
    # Explore dataset
    logger.info("\nStep 3: Exploring dataset...")
    
    loader = SpiderDataLoader(f"{args.data_dir}/spider")
    
    # Show samples
    logger.info(f"\n=== Sample Training Examples (n={args.samples}) ===")
    samples = loader.get_samples("train", args.samples)
    
    for i, sample in enumerate(samples, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Question: {sample['question']}")
        logger.info(f"  SQL: {sample.get('sql', 'N/A')}")
        logger.info(f"  Database: {sample['db_id']}")
        logger.info("-" * 50)
    
    # Show database info
    db_manager = SpiderDatabaseManager(f"{args.data_dir}/spider")
    databases = db_manager.list_available_databases()
    
    if databases:
        logger.info(f"\n=== Available Databases (n={len(databases)}) ===")
        logger.info(f"  {', '.join(databases[:10])}")
        if len(databases) > 10:
            logger.info(f"  ... and {len(databases) - 10} more")
    else:
        logger.info("\n⚠️  No physical database files available")
        logger.info("💡 Use --full flag to try downloading full dataset")
    
    logger.info("\n✅ Setup complete!")


if __name__ == "__main__":
    main()