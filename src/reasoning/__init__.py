"""
ReasoningBank: Self-Evolving Memory System for Text-to-SQL
"""

from .experience_collector import ExperienceCollector, Trajectory
from .self_judgment import SelfJudgment, JudgmentResult
from .strategy_distillation import StrategyDistillation, ReasoningStrategy
from .memory_store import ReasoningMemoryStore
from .memory_retrieval import MemoryRetrieval
from .memory_consolidation import MemoryConsolidation
from .test_time_scaling import ParallelScaling, SequentialScaling
from .reasoning_pipeline import ReasoningBankPipeline


__version__ = "1.0.0"

__all__ = [
    'ExperienceCollector',
    'Trajectory',
    'SelfJudgment',
    'JudgmentResult',
    'StrategyDistillation',
    'ReasoningStrategy',
    'ReasoningMemoryStore',
    'MemoryRetrieval',
    'MemoryConsolidation',
    'ParallelScaling',
    'SequentialScaling',
    'ReasoningBankPipeline',
]

# Check if dependencies are available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Memory retrieval will be limited.")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("Warning: SQLite not available. Memory persistence will be limited.")

def is_available() -> bool:
    """Check if ReasoningBank is fully available"""
    return CHROMADB_AVAILABLE and SQLITE_AVAILABLE


def create_reasoning_pipeline(
    db_path: str = "./memory/reasoning_bank.db",
    chromadb_path: str = "./memory/chromadb",
    config: dict = None
) -> ReasoningBankPipeline:
    """
    Convenience function to create a ReasoningBank pipeline
    
    Args:
        db_path: Path to SQLite database
        chromadb_path: Path to ChromaDB storage
        config: Configuration dictionary
        
    Returns:
        Initialized ReasoningBankPipeline
        
    Example:
        >>> pipeline = create_reasoning_pipeline()
        >>> result = pipeline.enhance_sql_generation(
        ...     question="How many cars?",
        ...     db_id="car_1",
        ...     schema=schema,
        ...     sql_generator=my_generator
        ... )
    """
    if not is_available():
        raise RuntimeError(
            "ReasoningBank dependencies not available. "
            "Install ChromaDB: pip install chromadb"
        )
    
    return ReasoningBankPipeline(
        db_path=db_path,
        chromadb_path=chromadb_path,
        config=config
    )