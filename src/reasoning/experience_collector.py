"""
Experience Collector: Captures SQL generation trajectories for learning

This module collects complete interaction trajectories including questions,
generated SQL, strategies used, and evaluation results.

COMPLETE FILE - Replace entire src/reasoning/experience_collector.py with this
"""

from dataclasses import dataclass, field  # ← IMPORTANT: field must be imported!
from typing import Dict, List, Optional
import time

# Logger
try:
    from utils.logging_utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """
    A single interaction trajectory capturing the SQL generation process
    
    Fixed: All non-default arguments come BEFORE default arguments
    """
    # ========================================
    # REQUIRED FIELDS (no defaults) - MUST BE FIRST
    # ========================================
    trajectory_id: str
    question: str
    database: str
    generated_sql: str
    
    # ========================================
    # OPTIONAL FIELDS (with defaults) - MUST BE AFTER required fields
    # ========================================
    schema: Optional[Dict] = None
    gold_sql: Optional[str] = None
    strategies_used: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # Evaluation results (filled after evaluation)
    exact_match: Optional[float] = None
    execution_match: Optional[bool] = None
    component_scores: Optional[Dict] = None
    
    # Additional fields for SelfJudgment
    errors: Optional[List[str]] = None
    execution_time: Optional[float] = None
    retrieved_strategies: Optional[List[Dict]] = None
    prompt_used: Optional[str] = None
    semantic_analysis: Optional[Dict] = None
    difficulty: Optional[str] = None
    
    # Additional metadata
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert trajectory to dictionary"""
        return {
            'trajectory_id': self.trajectory_id,
            'question': self.question,
            'database': self.database,
            'schema': self.schema,
            'generated_sql': self.generated_sql,
            'gold_sql': self.gold_sql,
            'strategies_used': self.strategies_used,
            'reasoning_steps': self.reasoning_steps,
            'generation_time': self.generation_time,
            'timestamp': self.timestamp,
            'exact_match': self.exact_match,
            'execution_match': self.execution_match,
            'component_scores': self.component_scores,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trajectory':
        """Create trajectory from dictionary"""
        return cls(**data)


class ExperienceCollector:
    """
    Collects and manages interaction trajectories
    
    This module captures the complete generation process for each query,
    including the question, generated SQL, strategies used, and results.
    """
    
    def __init__(self):
        """Initialize experience collector"""
        self.trajectories: Dict[str, Trajectory] = {}
        self.judgments: Dict[str, 'JudgmentResult'] = {}
        
        logger.info("ExperienceCollector initialized")
    
    def add_trajectory(self, trajectory: Trajectory):
        """
        Add a trajectory to the collection
        
        Args:
            trajectory: Trajectory object to add
        """
        self.trajectories[trajectory.trajectory_id] = trajectory
        logger.debug(f"Added trajectory: {trajectory.trajectory_id}")
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """
        Get a trajectory by ID
        
        Args:
            trajectory_id: Trajectory identifier
            
        Returns:
            Trajectory object or None if not found
        """
        return self.trajectories.get(trajectory_id)
    
    def get_all_trajectories(self) -> List[Trajectory]:
        """
        Get all trajectories
        
        Returns:
            List of all Trajectory objects
        """
        return list(self.trajectories.values())
    
    def get_recent_trajectories(self, limit: int = 100) -> List[Trajectory]:
        """
        Get most recent trajectories
        
        Args:
            limit: Maximum number of trajectories to return
            
        Returns:
            List of recent Trajectory objects
        """
        trajectories = sorted(
            self.trajectories.values(),
            key=lambda t: t.timestamp,
            reverse=True
        )
        return trajectories[:limit]
    
    def add_judgment(self, trajectory_id: str, judgment: 'JudgmentResult'):
        """
        Add judgment result for a trajectory
        
        Args:
            trajectory_id: Trajectory identifier
            judgment: JudgmentResult object
        """
        self.judgments[trajectory_id] = judgment
        logger.debug(f"Added judgment for trajectory: {trajectory_id}")
    
    def get_judgment(self, trajectory_id: str) -> Optional['JudgmentResult']:
        """
        Get judgment for a trajectory
        
        Args:
            trajectory_id: Trajectory identifier
            
        Returns:
            JudgmentResult object or None if not found
        """
        return self.judgments.get(trajectory_id)
    
    def create_trajectory(
        self,
        trajectory_id: str,
        question: str,
        database: str,
        generated_sql: str,
        schema: Optional[Dict] = None,
        gold_sql: Optional[str] = None,
        strategies_used: Optional[List[str]] = None,
        reasoning_steps: Optional[List[str]] = None,
        generation_time: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> Trajectory:
        """
        Convenience method to create and add a trajectory
        
        Args:
            trajectory_id: Unique identifier
            question: Natural language query
            database: Database name
            generated_sql: Generated SQL query
            schema: Database schema
            gold_sql: Gold standard SQL
            strategies_used: List of strategy IDs used
            reasoning_steps: Reasoning steps taken
            generation_time: Time taken to generate
            metadata: Additional metadata
            
        Returns:
            Created Trajectory object
        """
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            question=question,
            database=database,
            generated_sql=generated_sql,
            schema=schema,
            gold_sql=gold_sql,
            strategies_used=strategies_used or [],
            reasoning_steps=reasoning_steps or [],
            generation_time=generation_time,
            metadata=metadata or {}
        )
        
        self.add_trajectory(trajectory)
        return trajectory
    
    def clear(self):
        """Clear all trajectories and judgments"""
        self.trajectories.clear()
        self.judgments.clear()
        logger.info("Cleared all trajectories and judgments")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about collected trajectories
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.trajectories)
        judged = len(self.judgments)
        
        if judged > 0:
            successful = sum(
                1 for j in self.judgments.values()
                if j.is_success()
            )
            success_rate = successful / judged
        else:
            success_rate = 0.0
        
        avg_time = (
            sum(t.generation_time for t in self.trajectories.values()) / total
            if total > 0 else 0.0
        )
        
        return {
            'total_trajectories': total,
            'judged_trajectories': judged,
            'success_rate': success_rate,
            'avg_generation_time': avg_time
        }
    
    def __len__(self) -> int:
        """Return number of trajectories"""
        return len(self.trajectories)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ExperienceCollector(trajectories={len(self.trajectories)}, judgments={len(self.judgments)})"