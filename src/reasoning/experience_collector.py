"""
Experience Collector: Captures complete interaction trajectories
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Trajectory:
    """Complete interaction trajectory"""
    
    # Identifiers
    trajectory_id: str
    timestamp: str
    
    # Input
    query: str
    database: str
    schema: Dict
    difficulty: Optional[str] = None
    
    # Processing
    semantic_analysis: Optional[Dict] = None
    retrieved_examples: Optional[List[Dict]] = None
    retrieved_strategies: Optional[List[Dict]] = None
    prompt_used: Optional[str] = None
    
    # Output
    generated_sql: str
    generation_time: float = 0.0
    
    # Evaluation
    exact_match: bool = False
    execution_match: bool = False
    component_scores: Optional[Dict] = None
    execution_time: float = 0.0
    
    # Metadata
    errors: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ExperienceCollector:
    """Collects and stores interaction trajectories"""
    
    def __init__(self, storage_path: str = "./memory/trajectories"):
        """
        Initialize experience collector
        
        Args:
            storage_path: Path to store trajectory files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.trajectories = []
        self.current_trajectory = None
        
        logger.info(f"ExperienceCollector initialized: {storage_path}")
    
    def start_trajectory(
        self,
        query: str,
        database: str,
        schema: Dict,
        difficulty: Optional[str] = None
    ) -> str:
        """
        Start a new trajectory
        
        Args:
            query: Natural language question
            database: Database name
            schema: Database schema
            difficulty: Query difficulty level
            
        Returns:
            Trajectory ID
        """
        trajectory_id = f"traj_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.current_trajectory = Trajectory(
            trajectory_id=trajectory_id,
            timestamp=datetime.now().isoformat(),
            query=query,
            database=database,
            schema=schema,
            difficulty=difficulty,
            generated_sql="",
            errors=[]
        )
        
        logger.debug(f"Started trajectory: {trajectory_id}")
        return trajectory_id
    
    def add_semantic_analysis(self, analysis: Dict):
        """Add semantic analysis results"""
        if self.current_trajectory:
            self.current_trajectory.semantic_analysis = analysis
    
    def add_retrieved_examples(self, examples: List[Dict]):
        """Add retrieved examples from ChromaDB"""
        if self.current_trajectory:
            self.current_trajectory.retrieved_examples = examples
    
    def add_retrieved_strategies(self, strategies: List[Dict]):
        """Add retrieved strategies from ReasoningBank"""
        if self.current_trajectory:
            self.current_trajectory.retrieved_strategies = strategies
    
    def add_prompt(self, prompt: str):
        """Add the complete prompt used"""
        if self.current_trajectory:
            self.current_trajectory.prompt_used = prompt
    
    def add_generation_result(
        self,
        generated_sql: str,
        generation_time: float
    ):
        """Add SQL generation results"""
        if self.current_trajectory:
            self.current_trajectory.generated_sql = generated_sql
            self.current_trajectory.generation_time = generation_time
    
    def add_evaluation_results(
        self,
        exact_match: bool,
        execution_match: bool,
        component_scores: Dict,
        execution_time: float
    ):
        """Add evaluation results"""
        if self.current_trajectory:
            self.current_trajectory.exact_match = exact_match
            self.current_trajectory.execution_match = execution_match
            self.current_trajectory.component_scores = component_scores
            self.current_trajectory.execution_time = execution_time
    
    def add_error(self, error: str):
        """Add error message"""
        if self.current_trajectory:
            if self.current_trajectory.errors is None:
                self.current_trajectory.errors = []
            self.current_trajectory.errors.append(error)
    
    def finish_trajectory(self) -> Trajectory:
        """
        Finish and save current trajectory
        
        Returns:
            Completed trajectory
        """
        if not self.current_trajectory:
            raise ValueError("No active trajectory")
        
        # Save to memory
        self.trajectories.append(self.current_trajectory)
        
        # Save to disk
        self._save_trajectory(self.current_trajectory)
        
        completed = self.current_trajectory
        self.current_trajectory = None
        
        logger.info(f"Completed trajectory: {completed.trajectory_id}")
        return completed
    
    def _save_trajectory(self, trajectory: Trajectory):
        """Save trajectory to disk"""
        file_path = self.storage_path / f"{trajectory.trajectory_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                f.write(trajectory.to_json())
            logger.debug(f"Saved trajectory to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")
    
    def get_recent_trajectories(self, n: int = 10) -> List[Trajectory]:
        """Get N most recent trajectories"""
        return self.trajectories[-n:]
    
    def get_trajectories_by_success(self, success: bool) -> List[Trajectory]:
        """Get trajectories by success status"""
        return [
            t for t in self.trajectories
            if t.exact_match == success
        ]
    
    def get_statistics(self) -> Dict:
        """Get trajectory statistics"""
        if not self.trajectories:
            return {}
        
        total = len(self.trajectories)
        exact_match = sum(1 for t in self.trajectories if t.exact_match)
        execution_match = sum(1 for t in self.trajectories if t.execution_match)
        
        return {
            'total_trajectories': total,
            'exact_match_count': exact_match,
            'exact_match_rate': exact_match / total if total > 0 else 0,
            'execution_match_count': execution_match,
            'execution_match_rate': execution_match / total if total > 0 else 0,
            'avg_generation_time': sum(t.generation_time for t in self.trajectories) / total,
            'avg_execution_time': sum(t.execution_time for t in self.trajectories) / total
        }
    
    def load_trajectories_from_disk(self) -> int:
        """Load all trajectories from disk"""
        count = 0
        
        for file_path in self.storage_path.glob("traj_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trajectory = Trajectory(**data)
                    self.trajectories.append(trajectory)
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {count} trajectories from disk")
        return count