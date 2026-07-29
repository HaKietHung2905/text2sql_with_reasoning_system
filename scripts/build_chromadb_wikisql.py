"""
Script to build ChromaDB vector database from the WikiSQL dataset.

Mirrors scripts/build_chromadb.py (Spider), but uses WikiSQLChromaDBIntegration
and stores collections under the "wikisql" prefix so both datasets can
coexist in the same persist_dir.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.wikisql_chromadb_integration import WikiSQLChromaDBIntegration
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Build ChromaDB vector database from WikiSQL dataset'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./data/raw/wikisql',
        help='Directory containing WikiSQL train_spider_format.json / dev_spider_format.json '
             '(or raw train.json / dev.json, which will be auto-converted)'
    )
    parser.add_argument(
        '--persist-dir', type=str, default='./data/embeddings/chroma_db',
        help='Directory for ChromaDB storage (shared with Spider collections; '
             'names are namespaced by the "wikisql_" prefix)'
    )
    parser.add_argument(
        '--model', type=str, default='all-MiniLM-L6-v2',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--no-reset', action='store_true',
        help='Do not reset existing collections'
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WikiSQL Dataset + ChromaDB Integration Setup")
    logger.info("=" * 60)

    integration = WikiSQLChromaDBIntegration(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        model_name=args.model
    )

    logger.info("\nStep 1: Loading WikiSQL dataset...")
    if not integration.load_dataset():
        logger.error("Failed to load dataset")
        return 1

    logger.info("\nStep 2: Setting up ChromaDB collections...")
    integration.setup_collections(reset=not args.no_reset)

    logger.info("\nStep 3: Storing data in ChromaDB...")
    counts = integration.store_all_data()

    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nItems stored:")
    for key, count in counts.items():
        logger.info(f"  {key}: {count}")

    stats = integration.get_statistics()
    logger.info("\nCollection statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n✓ WikiSQL ChromaDB is ready for use!")
    return 0


if __name__ == "__main__":
    sys.exit(main())