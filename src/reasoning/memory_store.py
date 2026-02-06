import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Install with: pip install chromadb")

from .strategy_distillation import ReasoningStrategy
from utils.logging_utils import get_logger
from utils.embedding_utils import EmbeddingGenerator

logger = get_logger(__name__)


class ReasoningMemoryStore:
    """
    Persistent storage for reasoning strategies
    
    Features:
    - SQLite database for structured strategy data
    - ChromaDB for vector-based strategy retrieval
    - Version tracking for strategy evolution
    - Application history tracking
    - Performance metrics
    """
    
    def __init__(
        self,
        db_path: str = "./memory/reasoning_bank.db",
        chromadb_path: str = "./memory/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize reasoning memory store
        
        Args:
            db_path: Path to SQLite database
            chromadb_path: Path to ChromaDB storage
            embedding_model: Sentence transformer model name
        """
        self.db_path = Path(db_path)
        self.chromadb_path = Path(chromadb_path)
        
        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chromadb_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.db = self.conn  # Keep backward compatibility
        self._initialize_database_schema()
        
        # Initialize embedding generator
        self.embedding_gen = EmbeddingGenerator(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.strategy_collection = None
        
        if CHROMADB_AVAILABLE:
            self._initialize_chromadb(embedding_model)
        else:
            logger.warning("ChromaDB not available. Vector retrieval disabled.")
        
        logger.info(f"ReasoningMemoryStore initialized: {db_path}")
    
    def _initialize_chromadb(self, embedding_model: str):
        """Initialize ChromaDB for vector storage (using new API)"""
        
        try:
            # Use new PersistentClient API (v0.4.0+)
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chromadb_path)
            )
            
            # Create or get collection
            self.strategy_collection = self.chroma_client.get_or_create_collection(
                name="reasoning_strategies",
                metadata={
                    "description": "High-level SQL generation strategies",
                    "embedding_model": embedding_model,
                    "hnsw:space": "cosine"
                }
            )
            
            logger.info(f"ChromaDB initialized at: {self.chromadb_path}")
            logger.info(f"Collection count: {self.strategy_collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            logger.error(f"ChromaDB path: {self.chromadb_path}")
            self.chroma_client = None
            self.strategy_collection = None
        
    def _initialize_database_schema(self):
        """Create database tables if they don't exist"""
        
        # Strategies table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                pattern TEXT NOT NULL,
                description TEXT,
                reasoning_steps TEXT,      -- JSON array
                critical_rules TEXT,       -- JSON array
                sql_hints TEXT,            -- JSON object
                applicability TEXT,        -- JSON object
                common_pitfalls TEXT,      -- JSON array
                success_rate REAL DEFAULT 0.0,
                sample_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Strategy applications table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_applications (
                application_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                trajectory_id TEXT,
                query TEXT,
                database TEXT,
                difficulty TEXT,
                success BOOLEAN,
                exact_match REAL,
                execution_match REAL,
                generation_time REAL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            )
        """)
        
        # Strategy evolution table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_evolution (
                evolution_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                changes TEXT,
                performance_before REAL,
                performance_after REAL,
                performance_delta REAL,
                evolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            )
        """)
        
        # Performance metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id TEXT PRIMARY KEY,
                strategy_id TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            )
        """)
        
        # Create indices
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategies_pattern 
            ON strategies(pattern)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategies_success_rate 
            ON strategies(success_rate DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_applications_strategy 
            ON strategy_applications(strategy_id)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_applications_success 
            ON strategy_applications(success)
        """)
        
        self.conn.commit()
        logger.info("Database schema initialized")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        
        Supports multiple methods from EmbeddingGenerator
        """
        try:
            # Try different method names
            if hasattr(self.embedding_gen, 'encode'):
                return self.embedding_gen.encode(text).tolist()
            elif hasattr(self.embedding_gen, 'get_embedding'):
                return self.embedding_gen.get_embedding(text)
            elif hasattr(self.embedding_gen, 'embed'):
                return self.embedding_gen.embed(text)
            elif hasattr(self.embedding_gen, '__call__'):
                return self.embedding_gen(text)
            else:
                raise AttributeError(f"EmbeddingGenerator has no known embedding method")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Standard dimension for all-MiniLM-L6-v2
    
    def store_strategy(self, strategy: ReasoningStrategy) -> bool:
        """
        Store or update a reasoning strategy
        
        Args:
            strategy: ReasoningStrategy to store
            
        Returns:
            True if successful
        """
        try:
            # Check if strategy already exists
            existing = self._get_strategy_by_id(strategy.strategy_id)
            
            if existing:
                # Update existing strategy
                return self._update_strategy(strategy)
            else:
                # Insert new strategy
                return self._insert_strategy(strategy)
                
        except Exception as e:
            logger.error(f"Failed to store strategy {strategy.strategy_id}: {e}")
            return False
    
    def _insert_strategy(self, strategy: ReasoningStrategy) -> bool:
        """Insert new strategy"""
        
        try:
            self.conn.execute("""
                INSERT INTO strategies (
                    strategy_id, name, pattern, description,
                    reasoning_steps, critical_rules, sql_hints,
                    applicability, common_pitfalls,
                    success_rate, sample_count,
                    created_at, last_updated, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.strategy_id,
                strategy.name,
                strategy.pattern,
                strategy.description,
                json.dumps(strategy.reasoning_steps),
                json.dumps(strategy.critical_rules),
                json.dumps(strategy.sql_template_hints),
                json.dumps(strategy.applicability),
                json.dumps(strategy.common_pitfalls),
                strategy.success_rate,
                strategy.sample_count,
                strategy.created_at,
                strategy.last_updated,
                strategy.version
            ))
            
            self.conn.commit()
            
            # Store in ChromaDB for vector retrieval
            if self.strategy_collection:
                self._store_strategy_embedding(strategy)
            
            logger.info(f"Stored new strategy: {strategy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert strategy: {e}")
            self.conn.rollback()
            return False
    
    def _update_strategy(self, strategy: ReasoningStrategy) -> bool:
        """Update existing strategy"""
        
        try:
            # Get old version for evolution tracking
            old_strategy = self._get_strategy_by_id(strategy.strategy_id)
            old_success_rate = old_strategy['success_rate'] if old_strategy else 0.0
            
            # Update strategy
            self.conn.execute("""
                UPDATE strategies SET
                    name = ?,
                    pattern = ?,
                    description = ?,
                    reasoning_steps = ?,
                    critical_rules = ?,
                    sql_hints = ?,
                    applicability = ?,
                    common_pitfalls = ?,
                    success_rate = ?,
                    sample_count = ?,
                    last_updated = ?,
                    version = ?
                WHERE strategy_id = ?
            """, (
                strategy.name,
                strategy.pattern,
                strategy.description,
                json.dumps(strategy.reasoning_steps),
                json.dumps(strategy.critical_rules),
                json.dumps(strategy.sql_template_hints),
                json.dumps(strategy.applicability),
                json.dumps(strategy.common_pitfalls),
                strategy.success_rate,
                strategy.sample_count,
                datetime.now().isoformat(),
                strategy.version,
                strategy.strategy_id
            ))
            
            self.conn.commit()
            
            # Track evolution
            if old_success_rate != strategy.success_rate:
                self._track_evolution(
                    strategy.strategy_id,
                    strategy.version,
                    old_success_rate,
                    strategy.success_rate,
                    "Performance update"
                )
            
            # Update ChromaDB embedding
            if self.strategy_collection:
                self._update_strategy_embedding(strategy)
            
            logger.info(f"Updated strategy: {strategy.name} (v{strategy.version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update strategy: {e}")
            self.conn.rollback()
            return False
    
    def _store_strategy_embedding(self, strategy: ReasoningStrategy):
        """Store strategy embedding in ChromaDB"""
        
        if not self.strategy_collection:
            return
        
        try:
            # Generate text representation
            strategy_text = self._strategy_to_text(strategy)
            
            # Generate embedding
            embedding = self._get_embedding(strategy_text)
            
            # Store in ChromaDB
            self.strategy_collection.add(
                ids=[strategy.strategy_id],
                embeddings=[embedding],
                documents=[strategy_text],
                metadatas=[{
                    "name": strategy.name,
                    "pattern": strategy.pattern,
                    "success_rate": strategy.success_rate,
                    "sample_count": strategy.sample_count,
                    "version": strategy.version
                }]
            )
            
            logger.debug(f"Stored embedding for: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
    
    def _update_strategy_embedding(self, strategy: ReasoningStrategy):
        """Update strategy embedding in ChromaDB"""
        
        if not self.strategy_collection:
            return
        
        try:
            # Delete old embedding
            self.strategy_collection.delete(ids=[strategy.strategy_id])
            
            # Store new embedding
            self._store_strategy_embedding(strategy)
            
        except Exception as e:
            logger.error(f"Failed to update embedding: {e}")
    
    def _strategy_to_text(self, strategy: ReasoningStrategy) -> str:
        """Convert strategy to text for embedding"""
        
        text_parts = [
            f"Strategy: {strategy.name}",
            f"Pattern: {strategy.pattern}",
            f"Description: {strategy.description}",
            "",
            "Reasoning Steps:",
            *[f"  {step}" for step in strategy.reasoning_steps],
            "",
            "Critical Rules:",
            *[f"  {rule}" for rule in strategy.critical_rules],
            "",
            "Applicability:",
            f"  Keywords: {', '.join(strategy.applicability.get('keywords', []))}",
            f"  Intent Types: {', '.join(strategy.applicability.get('intent_types', []))}",
            f"  SQL Patterns: {', '.join(strategy.applicability.get('sql_patterns', []))}"
        ]
        
        return "\n".join(text_parts)
    
    def get_strategy(self, strategy_id: str) -> Optional[ReasoningStrategy]:
        """
        Retrieve a strategy by ID
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            ReasoningStrategy or None
        """
        row = self._get_strategy_by_id(strategy_id)
        
        if row:
            return self._row_to_strategy(row)
        return None
    
    def _get_strategy_by_id(self, strategy_id: str) -> Optional[sqlite3.Row]:
        """Get strategy row from database"""
        
        cursor = self.conn.execute(
            "SELECT * FROM strategies WHERE strategy_id = ?",
            (strategy_id,)
        )
        return cursor.fetchone()
    
    def _row_to_strategy(self, row: sqlite3.Row) -> ReasoningStrategy:
        """Convert database row to ReasoningStrategy"""
        
        return ReasoningStrategy(
            strategy_id=row['strategy_id'],
            name=row['name'],
            pattern=row['pattern'],
            description=row['description'],
            reasoning_steps=json.loads(row['reasoning_steps']),
            critical_rules=json.loads(row['critical_rules']),
            sql_template_hints=json.loads(row['sql_hints']),
            applicability=json.loads(row['applicability']),
            common_pitfalls=json.loads(row['common_pitfalls']),
            success_rate=row['success_rate'],
            sample_count=row['sample_count'],
            created_at=row['created_at'],
            last_updated=row['last_updated'],
            version=row['version']
        )
    
    def get_all_strategies(
        self,
        active_only: bool = True,
        min_success_rate: float = 0.0
    ) -> List[ReasoningStrategy]:
        """
        Get all strategies
        
        Args:
            active_only: Only return active strategies
            min_success_rate: Minimum success rate filter
            
        Returns:
            List of strategies
        """
        query = "SELECT * FROM strategies WHERE 1=1"
        params = []
        
        if active_only:
            query += " AND is_active = 1"
        
        if min_success_rate > 0:
            query += " AND success_rate >= ?"
            params.append(min_success_rate)
        
        query += " ORDER BY success_rate DESC"
        
        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_strategy(row) for row in rows]
    
    def get_strategies_by_pattern(self, pattern: str) -> List[ReasoningStrategy]:
        """Get strategies matching a specific pattern"""
        
        cursor = self.conn.execute(
            "SELECT * FROM strategies WHERE pattern = ? AND is_active = 1 ORDER BY success_rate DESC",
            (pattern,)
        )
        rows = cursor.fetchall()
        
        return [self._row_to_strategy(row) for row in rows]
    
    def search_strategies(
        self,
        query: str,
        n_results: int = 5,
        min_success_rate: float = 0.0
    ) -> List[Tuple[ReasoningStrategy, float]]:
        """
        Search strategies using vector similarity
        
        Args:
            query: Search query (natural language or SQL-like)
            n_results: Number of results to return
            min_success_rate: Minimum success rate filter
            
        Returns:
            List of (strategy, similarity_score) tuples
        """
        if not self.strategy_collection:
            logger.debug("ChromaDB not available. Using SQL-based fallback.")
            strategies = self.get_all_strategies(min_success_rate=min_success_rate)
            return [(s, 0.0) for s in strategies[:n_results]]
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Search ChromaDB
            results = self.strategy_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, 10),  # Get more candidates, but limit to 10
                where={"success_rate": {"$gte": min_success_rate}} if min_success_rate > 0 else None
            )
            
            # Convert to strategies with scores
            strategies_with_scores = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, strategy_id in enumerate(results['ids'][0]):
                    strategy = self.get_strategy(strategy_id)
                    
                    if strategy:
                        # Calculate similarity (1 - distance for cosine distance)
                        similarity = 1.0 - results['distances'][0][i]
                        strategies_with_scores.append((strategy, similarity))
            
            # If no results, fall back to SQL
            if not strategies_with_scores:
                logger.debug("No ChromaDB results, falling back to SQL")
                strategies = self.get_all_strategies(min_success_rate=min_success_rate)
                return [(s, 0.0) for s in strategies[:n_results]]
            
            # Sort by combined score (similarity + success rate)
            strategies_with_scores.sort(
                key=lambda x: (x[1] * 0.6 + x[0].success_rate * 0.4),
                reverse=True
            )
            
            return strategies_with_scores[:n_results]
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}", exc_info=True)
            # Fallback to SQL-based search
            strategies = self.get_all_strategies(min_success_rate=min_success_rate)
            return [(s, 0.0) for s in strategies[:n_results]]
    
    def record_application(
        self,
        strategy_id: str,
        trajectory_id: str,
        query: str,
        database: str,
        difficulty: str,
        success: bool,
        exact_match: float,
        execution_match: bool,
        generation_time: float
    ) -> bool:
        """
        Record when a strategy was applied
        """
        try:
            application_id = self._generate_id("app")
            
            self.conn.execute("""
                INSERT INTO strategy_applications (
                    application_id, strategy_id, trajectory_id,
                    query, database, difficulty,
                    success, exact_match, execution_match,
                    generation_time, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                application_id,
                strategy_id,
                trajectory_id,
                query,
                database,
                difficulty,
                success,
                exact_match,
                execution_match,
                generation_time,
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to record application: {e}")
            self.conn.rollback()
            return False
    
    def _track_evolution(
        self,
        strategy_id: str,
        version: int,
        performance_before: float,
        performance_after: float,
        changes: str
    ):
        """Track strategy evolution"""
        
        try:
            evolution_id = self._generate_id("evo")
            delta = performance_after - performance_before
            
            self.conn.execute("""
                INSERT INTO strategy_evolution (
                    evolution_id, strategy_id, version,
                    changes, performance_before, performance_after,
                    performance_delta, evolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evolution_id,
                strategy_id,
                version,
                changes,
                performance_before,
                performance_after,
                delta,
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to track evolution: {e}")
            self.conn.rollback()
    
    def get_strategy_history(self, strategy_id: str) -> List[Dict]:
        """Get evolution history for a strategy"""
        
        cursor = self.conn.execute("""
            SELECT * FROM strategy_evolution
            WHERE strategy_id = ?
            ORDER BY version ASC
        """, (strategy_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_strategy_performance(
        self,
        strategy_id: str,
        limit: int = 100
    ) -> Dict:
        """Get performance metrics for a strategy"""
        
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_applications,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                AVG(exact_match) as avg_exact_match,
                SUM(CASE WHEN execution_match = 1 THEN 1 ELSE 0 END) as execution_matches,
                AVG(generation_time) as avg_generation_time
            FROM strategy_applications
            WHERE strategy_id = ?
            ORDER BY applied_at DESC
            LIMIT ?
        """, (strategy_id, limit))
        
        row = cursor.fetchone()
        
        if row and row['total_applications'] > 0:
            total = row['total_applications']
            return {
                'total_applications': total,
                'success_rate': row['successes'] / total,
                'avg_exact_match': row['avg_exact_match'] or 0.0,
                'execution_match_rate': row['execution_matches'] / total,
                'avg_generation_time': row['avg_generation_time'] or 0.0
            }
        
        return {
            'total_applications': 0,
            'success_rate': 0.0,
            'avg_exact_match': 0.0,
            'execution_match_rate': 0.0,
            'avg_generation_time': 0.0
        }
    
    def get_statistics(self) -> Dict:
        """Get overall memory store statistics"""
        
        # Strategy counts
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_strategies,
                COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_strategies,
                AVG(success_rate) as avg_success_rate,
                SUM(sample_count) as total_samples
            FROM strategies
        """)
        row = cursor.fetchone()
        
        # Application counts
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_applications,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_applications
            FROM strategy_applications
        """)
        app_row = cursor.fetchone()
        
        # Pattern distribution
        cursor = self.conn.execute("""
            SELECT pattern, COUNT(*) as count
            FROM strategies
            WHERE is_active = 1
            GROUP BY pattern
            ORDER BY count DESC
        """)
        patterns = {row['pattern']: row['count'] for row in cursor.fetchall()}
        
        stats = {
            'total_strategies': row['total_strategies'],
            'active_strategies': row['active_strategies'],
            'avg_success_rate': row['avg_success_rate'] or 0.0,
            'total_samples': row['total_samples'] or 0,
            'total_applications': app_row['total_applications'],
            'successful_applications': app_row['successful_applications'],
            'pattern_distribution': patterns
        }
        
        # ChromaDB stats
        if self.strategy_collection:
            try:
                stats['chromadb_count'] = self.strategy_collection.count()
            except:
                stats['chromadb_count'] = 0
        
        return stats
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate a strategy (soft delete)"""
        
        try:
            self.conn.execute(
                "UPDATE strategies SET is_active = 0 WHERE strategy_id = ?",
                (strategy_id,)
            )
            self.conn.commit()
            logger.info(f"Deactivated strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate strategy: {e}")
            return False
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """Permanently delete a strategy"""
        
        try:
            # Delete from SQLite
            self.conn.execute("DELETE FROM strategies WHERE strategy_id = ?", (strategy_id,))
            self.conn.execute("DELETE FROM strategy_applications WHERE strategy_id = ?", (strategy_id,))
            self.conn.execute("DELETE FROM strategy_evolution WHERE strategy_id = ?", (strategy_id,))
            self.conn.commit()
            
            # Delete from ChromaDB
            if self.strategy_collection:
                try:
                    self.strategy_collection.delete(ids=[strategy_id])
                except:
                    pass
            
            logger.info(f"Deleted strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete strategy: {e}")
            self.conn.rollback()
            return False
    
    def export_strategies(self, output_path: str) -> bool:
        """Export all strategies to JSON file"""
        
        try:
            strategies = self.get_all_strategies(active_only=False)
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_strategies': len(strategies),
                'strategies': [s.to_dict() for s in strategies]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(strategies)} strategies to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export strategies: {e}")
            return False
    
    def import_strategies(self, input_path: str) -> int:
        """Import strategies from JSON file"""
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            count = 0
            for strategy_dict in data['strategies']:
                strategy = ReasoningStrategy(**strategy_dict)
                if self.store_strategy(strategy):
                    count += 1
            
            logger.info(f"Imported {count} strategies from {input_path}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to import strategies: {e}")
            return 0
    
    def _generate_id(self, prefix: str = "id") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def close(self):
        """Close database connections"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()