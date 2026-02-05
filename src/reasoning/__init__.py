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