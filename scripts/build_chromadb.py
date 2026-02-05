"""
Script to build ChromaDB vector database from Spider dataset.

This script loads the Spider dataset and stores it in ChromaDB
for efficient retrieval during Text-to-SQL generation.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.spider_chromadb_integration import SpiderChromaDBIntegration
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """Main setup script"""
    parser = argparse.ArgumentParser(
        description='Build ChromaDB vector database from Spider dataset'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/raw/spider',
        help='Directory containing Spider dataset'
    )
    
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./data/embeddings/chroma_db',
        help='Directory for ChromaDB storage'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset if not found locally'
    )
    
    parser.add_argument(
        '--no-reset',
        action='store_true',
        help='Do not reset existing collections'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Spider Dataset + ChromaDB Integration Setup")
    logger.info("=" * 60)
    
    # Initialize integration
    try:
        integration = SpiderChromaDBIntegration(
            data_dir=args.data_dir,
            persist_dir=args.persist_dir,
            model_name=args.model
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # Load dataset
    logger.info("\nStep 1: Loading Spider dataset...")
    if not integration.load_dataset(download_if_missing=args.download):
        logger.error("Failed to load dataset")
        return 1
    
    # Setup collections
    logger.info("\nStep 2: Setting up ChromaDB collections...")
    integration.setup_collections(reset=not args.no_reset)
    
    # Store data
    logger.info("\nStep 3: Storing data in ChromaDB...")
    counts = integration.store_all_data()
    
    # Show results
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 60)
    
    logger.info("\nItems stored:")
    for key, count in counts.items():
        logger.info(f"  {key}: {count}")
    
    # Show statistics
    stats = integration.get_statistics()
    logger.info("\nCollection statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n✓ ChromaDB is ready for use!")
    logger.info("💡 You can now run your Text-to-SQL system with retrieval")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())