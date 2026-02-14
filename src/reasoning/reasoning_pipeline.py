"""
ReasoningBank Pipeline: Complete integration for self-evolving memory system

This module provides the main pipeline that connects all ReasoningBank components
to the evaluation system, enabling continuous learning from experiences.
"""

import time
import uuid
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from src.reasoning import (
    ExperienceCollector,
    Trajectory,
    SelfJudgment,
    JudgmentResult,
    StrategyDistillation,
    ReasoningStrategy,
    ReasoningMemoryStore,
    MemoryRetrieval,
    MemoryConsolidation,
    ParallelScaling,
    SequentialScaling
)
from src.reasoning.memory_retrieval import RetrievalContext, RetrievalResult
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReasoningBankPipeline:
    """
    Complete ReasoningBank pipeline for self-evolving memory system
    
    Features:
    - Automatic trajectory collection during SQL generation
    - Self-judgment of success/failure
    - Strategy distillation from experiences
    - Memory consolidation and evolution
    - Test-time scaling for difficult queries
    """
    
    def __init__(
        self,
        db_path: str = "./memory/reasoning_bank.db",
        chromadb_path: str = "./memory/chromadb",
        config: Optional[Dict] = None
    ):
        """
        Initialize ReasoningBank pipeline
        
        Args:
            db_path: Path to SQLite database
            chromadb_path: Path to ChromaDB storage
            config: Configuration dictionary
        """
        self.config = self._default_config()
        if config:
            self.config.update(config)
        
        # Initialize core components
        logger.info("Initializing ReasoningBank Pipeline...")
        
        # Memory store
        self.memory_store = ReasoningMemoryStore(
            db_path=db_path,
            chromadb_path=chromadb_path
        )
        
        # Memory retrieval
        self.memory_retrieval = MemoryRetrieval(
            memory_store=self.memory_store
        )
        
        # Experience collector
        self.experience_collector = ExperienceCollector()
        
        # Self-judgment
        self.self_judgment = SelfJudgment()
        
        # Strategy distillation
        self.strategy_distillation = StrategyDistillation()
        
        # Memory consolidation
        self.memory_consolidation = MemoryConsolidation(
            memory_store=self.memory_store,
            distillation=self.strategy_distillation
        )
        
        # Test-time scaling (optional)
        self.parallel_scaling = None
        self.sequential_scaling = None
        
        # Statistics
        self.stats = {
            'trajectories_collected': 0,
            'strategies_distilled': 0,
            'strategies_used': 0,
            'memory_consolidations': 0,
            'test_time_scaling_used': 0
        }
        
        logger.info("✓ ReasoningBank Pipeline initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'enable_retrieval': True,
            'enable_distillation': True,
            'enable_consolidation': True,
            'enable_test_time_scaling': False,
            'consolidation_frequency': 50,  # Consolidate every N queries
            'min_trajectories_for_distillation': 10,
            'use_parallel_scaling_for_hard_queries': True,
            'parallel_scaling_candidates': 5
        }
    
    def enhance_sql_generation(
        self,
        question: str,
        db_id: str,
        schema: Dict,
        gold_sql: Optional[str] = None,
        semantic_analysis: Optional[Dict] = None,
        sql_generator: Optional[callable] = None
    ) -> Dict:
        """
        Enhance SQL generation with ReasoningBank
        
        Args:
            question: Natural language query
            db_id: Database identifier
            schema: Database schema
            gold_sql: Gold standard SQL (for learning)
            semantic_analysis: Semantic layer output
            sql_generator: Function to generate SQL
            
        Returns:
            Dictionary with generated SQL and metadata
        """
        trajectory_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Step 1: Retrieve relevant strategies from memory
        strategies_used = []
        if self.config['enable_retrieval']:
            strategies_used = self._retrieve_strategies(
                question=question,
                db_id=db_id,
                schema=schema,
                semantic_analysis=semantic_analysis
            )
        
        # Step 2: Determine if we should use test-time scaling
        use_scaling = self._should_use_test_time_scaling(
            semantic_analysis=semantic_analysis,
            strategies_used=strategies_used
        )
        
        # Step 3: Generate SQL
        if use_scaling and sql_generator:
            result = self._generate_with_scaling(
                question=question,
                db_id=db_id,
                schema=schema,
                strategies_used=strategies_used,
                sql_generator=sql_generator
            )
            predicted_sql = result['sql']
            generation_metadata = result['metadata']
        else:
            # Standard generation
            predicted_sql = sql_generator(question) if sql_generator else None
            generation_metadata = {
                'method': 'standard',
                'strategies_applied': len(strategies_used)
            }
        
        generation_time = time.time() - start_time
        
        # Step 4: Create trajectory for learning
        trajectory = self._create_trajectory(
            trajectory_id=trajectory_id,
            question=question,
            db_id=db_id,
            schema=schema,
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            strategies_used=strategies_used,
            semantic_analysis=semantic_analysis,
            generation_time=generation_time,
            generation_metadata=generation_metadata
        )
        
        # Step 5: Collect trajectory
        self.experience_collector.add_trajectory(trajectory)
        self.stats['trajectories_collected'] += 1
        
        return {
            'sql': predicted_sql,
            'trajectory_id': trajectory_id,
            'strategies_used': [s.name for s in strategies_used],
            'generation_time': generation_time,
            'metadata': generation_metadata
        }
    def _apply_strategies_to_prompt(
        self,
        base_prompt: str,
        strategies: List[ReasoningStrategy]
    ) -> str:
        """Apply retrieved strategies to enhance prompt"""
        enhanced_prompt = base_prompt
        for strategy in strategies:
            enhanced_prompt += f"\n\nStrategy: {strategy.name}\n{strategy.description}"
        return enhanced_prompt
    
    
    def process_evaluation_result(
        self,
        trajectory_id: str,
        exact_match: float,
        execution_match: bool,
        component_scores: Optional[Dict] = None
    ):
        """
        Process evaluation result and update memory
        
        Args:
            trajectory_id: Trajectory identifier
            exact_match: Exact match score
            execution_match: Execution match result
            component_scores: Component-level scores
        """
        try:
            # Get trajectory
            trajectory = self.experience_collector.get_trajectory(trajectory_id)
            if not trajectory:
                logger.warning(f"Trajectory not found: {trajectory_id}")
                return
            
            # Update trajectory with results
            trajectory.exact_match = exact_match
            trajectory.execution_match = execution_match
            trajectory.component_scores = component_scores
            
            # Self-judgment
            judgment = self.self_judgment.judge_trajectory(trajectory)
            
            # Store judgment
            self.experience_collector.add_judgment(trajectory_id, judgment)
            
            # Record strategy applications
            for strategy_id in trajectory.strategies_used:
                try:
                    # FIXED: Safely get metadata
                    difficulty = 'unknown'
                    if hasattr(trajectory, 'metadata') and trajectory.metadata:
                        difficulty = trajectory.metadata.get('difficulty', 'unknown')
                    elif hasattr(trajectory, 'difficulty'):
                        difficulty = trajectory.difficulty or 'unknown'
                    
                    self.memory_store.record_application(
                        strategy_id=strategy_id,
                        trajectory_id=trajectory_id,
                        query=trajectory.question,
                        database=trajectory.database,
                        difficulty=difficulty,
                        success=judgment.is_success(),
                        exact_match=exact_match,
                        execution_match=execution_match,
                        generation_time=trajectory.generation_time
                    )
                except Exception as e:
                    logger.debug(f"Failed to record application for {strategy_id}: {e}")
            
            # Check if we should consolidate
            if self._should_consolidate():
                try:
                    self.consolidate_memory()
                except Exception as e:
                    logger.warning(f"Consolidation failed (non-critical): {e}")
        
        except Exception as e:
            logger.error(f"Failed to process evaluation result for {trajectory_id}: {e}")

    
    
    def consolidate_memory(self):
        """
        Consolidate memory: distill new strategies and refine existing ones
        
        FULLY SAFE VERSION - Always returns a dict
        """
        logger.info("🧠 Consolidating memory...")
        
        # CRITICAL: Initialize result at the VERY BEGINNING
        result = {
            'new_strategies': 0,
            'updated_strategies': 0,
            'patterns_identified': 0,
            'total_trajectories': 0,
            'errors': []
        }
        
        try:
            # Get recent trajectories
            trajectories = self.experience_collector.get_all_trajectories()
            result['total_trajectories'] = len(trajectories)
            
            if len(trajectories) < self.config.get('min_trajectories_for_distillation', 10):
                logger.info(f"Not enough trajectories for distillation: {len(trajectories)}")
                return result
            
            # Distill new strategies from successful trajectories
            if self.config.get('enable_distillation', True):
                logger.info("  Distilling strategies...")
                
                try:
                    # Get judgments for trajectories
                    judgments = []
                    successful_trajectories = []
                    
                    for t in trajectories:
                        judgment = self.experience_collector.get_judgment(t.trajectory_id)
                        if judgment and judgment.is_success():
                            successful_trajectories.append(t)
                            judgments.append(judgment)
                    
                    logger.info(f"  Found {len(successful_trajectories)} successful trajectories")
                    
                    if successful_trajectories and len(successful_trajectories) >= 3:
                        # Distill strategies
                        new_strategies = self.strategy_distillation.distill_strategies(
                            trajectories=successful_trajectories,
                            judgments=judgments
                        )
                        
                        # Store new strategies
                        for strategy in new_strategies:
                            if self.memory_store.store_strategy(strategy):
                                self.stats['strategies_distilled'] += 1
                                result['new_strategies'] += 1
                        
                        logger.info(f"✓ Distilled {len(new_strategies)} new strategies")
                        
                except Exception as e:
                    logger.error(f"Strategy distillation failed: {e}")
                    result['errors'].append(f"Distillation: {str(e)}")
            
            # Consolidate existing strategies
            if self.config.get('enable_consolidation', False):
                logger.info("  Consolidating existing strategies...")
                
                try:
                    # Get judgments for all trajectories
                    all_judgments = []
                    all_trajectories = []
                    
                    for t in trajectories:
                        judgment = self.experience_collector.get_judgment(t.trajectory_id)
                        if judgment is not None:
                            all_trajectories.append(t)
                            all_judgments.append(judgment)
                    
                    if all_trajectories:
                        consolidation_result = self.memory_consolidation.consolidate_memory(
                            new_trajectories=all_trajectories,
                            new_judgments=all_judgments
                        )
                        
                        self.stats['memory_consolidations'] += 1
                        
                        # Extract statistics safely
                        if consolidation_result and isinstance(consolidation_result, dict):
                            result['updated_strategies'] = consolidation_result.get('refined', 0)
                            result['patterns_identified'] = consolidation_result.get('patterns_discovered', 0)
                            
                            logger.info(f"✓ Memory consolidation complete")
                        
                except Exception as e:
                    logger.warning(f"Memory consolidation failed: {e}")
                    result['errors'].append(f"Consolidation: {str(e)}")
        
        except Exception as e:
            logger.error(f"Critical error in consolidate_memory: {e}")
            result['errors'].append(f"Critical: {str(e)}")
        
        return result
    
    def _retrieve_strategies(
        self,
        question: str,
        db_id: str,
        schema: Dict,
        semantic_analysis: Optional[Dict]
    ) -> List[ReasoningStrategy]:
        """Retrieve relevant strategies from memory"""
        
        retrieval_context = RetrievalContext(
            query=question,
            database=db_id,
            schema=schema,
            semantic_analysis=semantic_analysis
        )
        
        retrieval_results = self.memory_retrieval.retrieve_strategies(
            context=retrieval_context,
            top_k=5
        )
        
        strategies = [r.strategy for r in retrieval_results]
        self.stats['strategies_used'] += len(strategies)
        
        return strategies
    
    def _should_use_test_time_scaling(
        self,
        semantic_analysis: Optional[Dict],
        strategies_used: List[ReasoningStrategy]
    ) -> bool:
        """Determine if test-time scaling should be used"""
        
        if not self.config['enable_test_time_scaling']:
            return False
        
        # Use scaling for hard queries
        if semantic_analysis and semantic_analysis.get('complexity') == 'hard':
            return True
        
        # Use scaling if no good strategies found
        if len(strategies_used) == 0:
            return True
        
        # Use scaling if strategy confidence is low
        if strategies_used and all(s.success_rate < 0.6 for s in strategies_used):
            return True
        
        return False
    
    def _generate_with_scaling(
        self,
        question: str,
        db_id: str,
        schema: Dict,
        strategies_used: List[ReasoningStrategy],
        sql_generator: callable
    ) -> Dict:
        """Generate SQL with test-time scaling"""
        
        # Initialize parallel scaling if not done
        if not self.parallel_scaling:
            self.parallel_scaling = ParallelScaling(
                memory_store=self.memory_store,
                memory_retrieval=self.memory_retrieval,
                sql_generator=sql_generator
            )
        
        # Generate multiple candidates
        scaling_result = self.parallel_scaling.scale_generation(
            question=question,
            db_path=f"./data/database/{db_id}/{db_id}.sqlite",
            schema=schema,
            num_candidates=self.config['parallel_scaling_candidates'],
            strategies=strategies_used
        )
        
        self.stats['test_time_scaling_used'] += 1
        
        return {
            'sql': scaling_result.best_candidate.sql,
            'metadata': {
                'method': 'parallel_scaling',
                'num_candidates': len(scaling_result.all_candidates),
                'best_score': scaling_result.best_candidate.score,
                'total_time': scaling_result.total_time
            }
        }
    
    def _create_trajectory(
        self,
        trajectory_id: str,
        question: str,
        db_id: str,
        schema: Dict,
        predicted_sql: Optional[str],
        gold_sql: Optional[str],
        strategies_used: List[ReasoningStrategy],
        semantic_analysis: Optional[Dict],
        generation_time: float,
        generation_metadata: Dict
    ) -> Trajectory:
        """Create trajectory object"""
        
        return Trajectory(
            trajectory_id=trajectory_id,
            question=question,
            database=db_id,
            schema=schema,
            generated_sql=predicted_sql,
            gold_sql=gold_sql,
            strategies_used=[s.strategy_id for s in strategies_used],
            reasoning_steps=[s.reasoning_steps for s in strategies_used],
            generation_time=generation_time,
            timestamp=time.time(),
            metadata={
                'semantic_analysis': semantic_analysis,
                'generation_metadata': generation_metadata,
                'difficulty': semantic_analysis.get('complexity') if semantic_analysis else 'unknown'
            }
        )
    
    def _should_consolidate(self) -> bool:
        """Check if memory consolidation should run"""
        return (
            self.stats['trajectories_collected'] % 
            self.config['consolidation_frequency'] == 0
        )
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        return {
            **self.stats,
            'total_strategies_in_memory': len(self.memory_store.get_all_strategies()),
            'memory_store_stats': self.memory_store.get_statistics()
        }
    
    def save_statistics(self, output_path: str):
        """Save statistics to file"""
        stats = self.get_statistics()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to: {output_path}")

    def _validate_evaluation_result(
        self,
        exact_match: float,
        execution_match: bool
    ) -> bool:
        """Validate evaluation metrics are in expected ranges"""
        if not (0.0 <= exact_match <= 1.0):
            logger.warning(f"Invalid exact_match: {exact_match}")
            return False
        return True

    def update_trajectory(
        self,
        trajectory_id: str,
        success: bool,
        execution_success: bool = None,
        metadata: Optional[Dict] = None
    ):
        """Update trajectory with evaluation results"""
        try:
            exact_match = 1.0 if success else 0.0
            execution_match = execution_success if execution_success is not None else success
            component_scores = metadata.get('component_scores') if metadata else None
            
            return self.process_evaluation_result(
                trajectory_id=trajectory_id,
                exact_match=exact_match,
                execution_match=execution_match,
                component_scores=component_scores
            )
        except Exception as e:
            logger.error(f"Failed to update trajectory {trajectory_id}: {e}")
            return None

    def distill_strategies(self) -> Dict:
        """Distill new strategies from collected trajectories"""
        logger.info("🧠 Distilling strategies from trajectories...")
        
        result = {'new_strategies': 0, 'trajectories_processed': 0}
        
        try:
            trajectories = self.experience_collector.get_all_trajectories()
            result['trajectories_processed'] = len(trajectories)
            
            if len(trajectories) < self.config.get('min_trajectories_for_distillation', 10):
                return result
            
            successful_trajectories = []
            judgments = []
            
            for t in trajectories:
                judgment = self.experience_collector.get_judgment(t.trajectory_id)
                if judgment and judgment.is_success():
                    successful_trajectories.append(t)
                    judgments.append(judgment)
            
            if len(successful_trajectories) >= 3:
                new_strategies = self.strategy_distillation.distill_strategies(
                    trajectories=successful_trajectories,
                    judgments=judgments
                )
                
                for strategy in new_strategies:
                    if self.memory_store.store_strategy(strategy):
                        result['new_strategies'] += 1
                        self.stats['strategies_distilled'] += 1
        
        except Exception as e:
            result['error'] = str(e)
        
        return result