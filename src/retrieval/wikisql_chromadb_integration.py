"""
WikiSQL dataset integration with ChromaDB.

"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.retrieval.chromadb_handler import ChromaDBHandler
from utils.embedding_utils import EmbeddingGenerator
from utils.logging_utils import get_logger

logger = get_logger(__name__)

COLLECTION_PREFIX = "wikisql"

# Matches a bare "col" token, e.g. "WHERE col = 13" or "SELECT MIN(col)".
# Real WikiSQL column names in converted SQL are proper header strings
# (e.g. "Air_Date", "2001_Census") — a standalone lowercase "col" token is
# the unresolved-placeholder bug, not a legitimate column name.
_UNRESOLVED_COL_RE = re.compile(r'\bcol\b')


class WikiSQLChromaDBIntegration:
    """Integrate WikiSQL dataset with ChromaDB for Semantic RAG retrieval."""

    def __init__(
        self,
        data_dir: str = "./data/raw/wikisql",
        persist_dir: str = "./data/embeddings/chroma_db",
        model_name: str = None
    ):
        """
        Args:
            data_dir: Directory containing WikiSQL spider-format JSON files
                (train_spider_format.json / dev_spider_format.json), each a
                list of {"question": str, "db_id": str, "query"/"sql": str}.
            persist_dir: Directory for ChromaDB storage.
            model_name: Embedding model name.
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)

        self.db_handler = ChromaDBHandler(str(self.persist_dir))
        self.embedding_gen = EmbeddingGenerator(model_name)

        self.train_data: Optional[List[Dict]] = None
        self.dev_data: Optional[List[Dict]] = None

    def _load_spider_format_split(self, split: str) -> List[Dict]:
        """
        Load a WikiSQL split already converted to spider-format JSON
        (see scripts/evaluate_wikisql.py: convert_wikisql_gold_to_spider_format).
        Auto-converts from the raw WikiSQL file if the spider-format file
        is missing but the raw file is present.
        """
        spider_format_path = self.data_dir / f"{split}_spider_format.json"
        raw_path = self.data_dir / f"{split}.json"

        if not spider_format_path.exists() and raw_path.exists():
            logger.info(f"{spider_format_path} not found — converting from {raw_path}")
            try:
                from scripts.evaluate_wikisql import convert_wikisql_gold_to_spider_format
                convert_wikisql_gold_to_spider_format(
                    gold_file=str(raw_path),
                    output_file=str(spider_format_path),
                )
            except Exception as e:
                logger.error(f"Failed to auto-convert {split} split: {e}")
                return []

        if not spider_format_path.exists():
            logger.warning(f"No data found for split '{split}' at {spider_format_path}")
            return []

        with open(spider_format_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _is_unresolved_sql(sql: str) -> bool:
        """
        Return True if `sql` still contains an unresolved "col" placeholder
        left over from a failed WikiSQL column-index-to-name conversion.
        """
        if not sql:
            return True
        return bool(_UNRESOLVED_COL_RE.search(sql))

    def _filter_valid_examples(self, examples: List[Dict], split: str) -> Tuple[List[Dict], int]:
        """
        Drop examples whose gold SQL is still unresolved. Returns
        (valid_examples, skipped_count).
        """
        valid = []
        skipped = 0
        for example in examples:
            sql = example.get('query', example.get('sql', ''))
            if self._is_unresolved_sql(sql):
                skipped += 1
                continue
            valid.append(example)

        if skipped:
            pct = 100.0 * skipped / max(len(examples), 1)
            logger.warning(
                f"[{split}] Skipped {skipped}/{len(examples)} ({pct:.1f}%) examples "
                f"with unresolved 'col' placeholders — not embedded into ChromaDB."
            )
        return valid, skipped

    def load_dataset(self) -> bool:
        """Load WikiSQL train and dev splits (spider-format), filtering out
        examples with unresolved 'col' placeholders."""
        logger.info("Loading WikiSQL dataset...")

        raw_train = self._load_spider_format_split("train")
        raw_dev = self._load_spider_format_split("dev")

        self.train_data, train_skipped = self._filter_valid_examples(raw_train, "train")
        self.dev_data, dev_skipped = self._filter_valid_examples(raw_dev or [], "dev")

        if not self.train_data:
            logger.error(
                "No usable WikiSQL train data found (after filtering unresolved "
                f"placeholders). Expected {self.data_dir / 'train_spider_format.json'} "
                f"or {self.data_dir / 'train.json'}"
            )
            return False

        logger.info(
            f"Dataset loaded: Train={len(self.train_data)} "
            f"(dropped {train_skipped} unresolved), "
            f"Dev={len(self.dev_data)} (dropped {dev_skipped} unresolved)"
        )
        return True

    def setup_collections(self, reset: bool = True) -> None:
        """Setup ChromaDB collections under the 'wikisql' prefix."""
        logger.info("Setting up WikiSQL ChromaDB collections...")
        self.db_handler.setup_collections(reset=reset, prefix=COLLECTION_PREFIX)

    def _batch_encode(self, texts: List[str], desc: str, chunk_size: int = 256) -> List[List[float]]:
        """
        Encode many texts efficiently. Tries the underlying model's native
        batch encode() first (much faster than one-at-a-time on CPU); falls
        back to per-item encoding if the wrapper doesn't support a list
        input. Shows a tqdm progress bar either way, since embedding tens
        of thousands of questions can take a long time with no other
        visible output.
        """
        embeddings: List[List[float]] = []
        try:
            # Attempt native batch encoding, chunked to bound memory use.
            for start in tqdm(range(0, len(texts), chunk_size), desc=desc, unit="batch"):
                chunk = texts[start:start + chunk_size]
                batch_emb = self.embedding_gen.encode(chunk)
                # Normalize to a list of lists regardless of numpy/list return type.
                embeddings.extend([e.tolist() if hasattr(e, "tolist") else list(e) for e in batch_emb])
            if len(embeddings) == len(texts):
                return embeddings
            logger.warning("Batch encode returned unexpected length, falling back to per-item encoding")
            embeddings = []
        except Exception as e:
            logger.debug(f"Batch encode not supported/failed ({e}), falling back to per-item encoding")
            embeddings = []

        for text in tqdm(texts, desc=f"{desc} (per-item fallback)", unit="item"):
            emb = self.embedding_gen.encode(text)
            embeddings.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
        return embeddings

    def store_questions(self, examples: List[Dict], split: str) -> int:
        """Store raw question embeddings (no schema prefix — see module docstring).

        Examples are assumed to already be filtered (see load_dataset /
        _filter_valid_examples) — this method does not re-check for
        unresolved placeholders.
        """
        if not examples:
            logger.warning(f"No {split} data to store")
            return 0

        logger.info(f"Storing {split} questions ({len(examples)} examples)...")

        documents, ids, metadatas, texts_to_encode = [], [], [], []

        for i, example in enumerate(examples):
            question = example.get('question', '')
            if not question:
                continue
            sql = example.get('query', example.get('sql', ''))
            db_id = example.get('db_id', 'unknown')

            texts_to_encode.append(question)
            documents.append(question)
            ids.append(f"{split}_{db_id}_{i}")
            metadatas.append({
                "db_id": db_id,
                "question": question,
                "sql": sql,
                "split": split,
                "type": "question"
            })

        embeddings = self._batch_encode(texts_to_encode, desc=f"Embedding {split} questions")

        count = self.db_handler.add_batch(
            self.db_handler.question_collection,
            documents, embeddings, ids, metadatas,
            batch_size=1000
        )
        logger.info(f"Stored {count} {split} questions")
        return count


    def store_sql_queries(self, examples: List[Dict], split: str) -> int:
        """Store SQL query embeddings (mirrors Spider's sql_collection).

        Examples are assumed to already be filtered — see store_questions.
        """
        if not examples:
            return 0

        logger.info(f"Storing {split} SQL queries ({len(examples)} examples)...")

        documents, ids, metadatas, texts_to_encode = [], [], [], []

        for i, example in enumerate(examples):
            sql = example.get('query', example.get('sql', ''))
            question = example.get('question', '')
            db_id = example.get('db_id', 'unknown')
            if not sql:
                continue

            texts_to_encode.append(sql)
            documents.append(sql)
            ids.append(f"{split}_sql_{db_id}_{i}")
            metadatas.append({
                "db_id": db_id,
                "question": question,
                "sql": sql,
                "split": split,
                "type": "sql"
            })

        embeddings = self._batch_encode(texts_to_encode, desc=f"Embedding {split} SQL")

        count = self.db_handler.add_batch(
            self.db_handler.sql_collection,
            documents, embeddings, ids, metadatas,
            batch_size=1000
        )
        logger.info(f"Stored {count} {split} SQL queries")
        return count

    def store_all_data(self) -> Dict[str, int]:
        """
        Store all data. Note: no schema collection is populated — see
        module docstring for why WikiSQL schema retrieval isn't meaningful.
        Examples with unresolved 'col' placeholders were already dropped
        in load_dataset().
        """
        counts = {}
        counts['train_questions'] = self.store_questions(self.train_data, "train")
        counts['dev_questions'] = self.store_questions(self.dev_data or [], "dev")
        counts['train_sql'] = self.store_sql_queries(self.train_data, "train")
        counts['dev_sql'] = self.store_sql_queries(self.dev_data or [], "dev")
        return counts

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.db_handler.get_statistics()
        stats['train_examples'] = len(self.train_data) if self.train_data else 0
        stats['dev_examples'] = len(self.dev_data) if self.dev_data else 0
        return stats