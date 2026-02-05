"""
Test-Time Scaling: Memory-Aware Test-Time Scaling (MaTTS)

This module implements parallel and sequential scaling for SQL generation,
guided by reasoning strategies from the memory bank.

Based on ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from .memory_store import ReasoningMemoryStore
from .memory_retrieval import MemoryRetrieval, RetrievalContext, RetrievalResult
from .strategy_distillation import ReasoningStrategy
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Candidate:
    """A single SQL candidate"""
    candidate_id: str
    sql: str
    strategy_used: Optional[str]
    generation_method: str  # "parallel", "sequential", "refinement"
    temperature: float
    generation_time: float
    confidence: float = 0.0
    score: float = 0.0
    metadata: Optional[Dict] = None


@dataclass
class ScalingResult:
    """Result of test-time scaling"""
    best_candidate: Candidate
    all_candidates: List[Candidate]
    total_time: float
    num_candidates: int
    scaling_method: str
    improvement_metrics: Dict


class ParallelScaling:
    """
    Parallel test-time scaling: Generate multiple candidates simultaneously
    
    Features:
    - Memory-guided diverse generation
    - Strategy-based candidate generation
    - Temperature variation
    - Concurrent execution
    - Best candidate selection
    """
    
    def __init__(
        self,
        memory_store: ReasoningMemoryStore,
        memory_retrieval: MemoryRetrieval,
        sql_generator: Callable,
        max_workers: int = 5
    ):
        """
        Initialize parallel scaling
        
        Args:
            memory_store: ReasoningMemoryStore instance
            memory_retrieval: MemoryRetrieval instance
            sql_generator: Function to generate SQL (prompt -> SQL)
            max_workers: Maximum parallel workers
        """
        self.memory_store = memory_store
        self.memory_retrieval = memory_retrieval
        self.sql_generator = sql_generator
        self.max_workers = max_workers
        
        logger.info(f"ParallelScaling initialized with {max_workers} workers")
    
    def generate_diverse_candidates(
        self,
        context: RetrievalContext,
        n_candidates: int = 5,
        enable_memory_guidance: bool = True
    ) -> List[Candidate]:
        """
        Generate multiple diverse SQL candidates in parallel
        
        Args:
            context: Retrieval context with query and schema
            n_candidates: Number of candidates to generate
            enable_memory_guidance: Use retrieved strategies
            
        Returns:
            List of candidate SQL queries
        """
        logger.info(f"Generating {n_candidates} diverse candidates in parallel")
        start_time = time.time()
        
        # Step 1: Retrieve strategies if memory-guided
        strategies = []
        if enable_memory_guidance:
            retrieval_results = self.memory_retrieval.retrieve_strategies(
                context=context,
                top_k=min(n_candidates, 5),
                min_confidence=0.4
            )
            strategies = [r.strategy for r in retrieval_results]
        
        # Step 2: Create generation tasks
        tasks = self._create_generation_tasks(
            context,
            strategies,
            n_candidates
        )
        
        # Step 3: Execute tasks in parallel
        candidates = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_task, task): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    candidate = future.result()
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(candidates)} candidates in {total_time:.2f}s")
        
        return candidates
    
    def _create_generation_tasks(
        self,
        context: RetrievalContext,
        strategies: List[ReasoningStrategy],
        n_candidates: int
    ) -> List[Dict]:
        """Create diverse generation tasks"""
        
        tasks = []
        
        # Task type 1: Strategy-focused generation
        for i, strategy in enumerate(strategies[:n_candidates // 2]):
            tasks.append({
                'task_id': f"strategy_{i}",
                'type': 'strategy_focused',
                'context': context,
                'strategy': strategy,
                'temperature': 0.1 + (i * 0.1),  # Gradually increase
                'prompt_style': 'focused'
            })
        
        # Task type 2: Combined strategies
        if len(strategies) >= 2:
            for i in range(n_candidates // 3):
                # Use 2-3 strategies together
                strategy_subset = strategies[i:i+3] if i+3 <= len(strategies) else strategies[:3]
                tasks.append({
                    'task_id': f"combined_{i}",
                    'type': 'combined_strategies',
                    'context': context,
                    'strategies': strategy_subset,
                    'temperature': 0.2 + (i * 0.1),
                    'prompt_style': 'combined'
                })
        
        # Task type 3: Exploration (higher temperature)
        remaining = n_candidates - len(tasks)
        for i in range(remaining):
            tasks.append({
                'task_id': f"explore_{i}",
                'type': 'exploration',
                'context': context,
                'strategy': strategies[0] if strategies else None,
                'temperature': 0.5 + (i * 0.1),
                'prompt_style': 'exploratory'
            })
        
        return tasks[:n_candidates]
    
    def _execute_task(self, task: Dict) -> Optional[Candidate]:
        """Execute a single generation task"""
        
        start_time = time.time()
        
        try:
            # Build prompt based on task type
            prompt = self._build_prompt_for_task(task)
            
            # Generate SQL
            sql = self.sql_generator(
                prompt=prompt,
                temperature=task['temperature'],
                max_tokens=512
            )
            
            generation_time = time.time() - start_time
            
            # Create candidate
            candidate = Candidate(
                candidate_id=self._generate_id(task['task_id']),
                sql=sql,
                strategy_used=task.get('strategy', {}).get('name') if task.get('strategy') else 'none',
                generation_method='parallel',
                temperature=task['temperature'],
                generation_time=generation_time,
                metadata={
                    'task_type': task['type'],
                    'prompt_style': task['prompt_style']
                }
            )
            
            return candidate
            
        except Exception as e:
            logger.error(f"Failed to execute task {task['task_id']}: {e}")
            return None
    
    def _build_prompt_for_task(self, task: Dict) -> str:
        """Build prompt based on task configuration"""
        
        context = task['context']
        base_prompt = f"""
Database Schema:
{self._format_schema(context.schema)}

Question: {context.query}
"""
        
        if task['type'] == 'strategy_focused':
            # Focus on single strategy
            strategy = task['strategy']
            strategy_section = f"""
Strategy: {strategy.name}
Success Rate: {strategy.success_rate:.1%}

Reasoning Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(strategy.reasoning_steps))}

Critical Rules:
{chr(10).join(f"- {rule}" for rule in strategy.critical_rules)}
"""
            return base_prompt + strategy_section + "\nGenerate SQL:"
        
        elif task['type'] == 'combined_strategies':
            # Combine multiple strategies
            strategies = task['strategies']
            combined_section = f"""
Using {len(strategies)} complementary strategies:

{chr(10).join(f"Strategy {i+1}: {s.name} ({s.success_rate:.1%})" for i, s in enumerate(strategies))}

Key insights:
{chr(10).join(f"- {s.reasoning_steps[0]}" for s in strategies if s.reasoning_steps)}
"""
            return base_prompt + combined_section + "\nGenerate SQL:"
        
        else:  # exploration
            # Minimal guidance, encourage creativity
            return base_prompt + """
Generate a SQL query to answer the question.
Be creative and consider multiple approaches.

Generate SQL:"""
    
    def _format_schema(self, schema: Dict) -> str:
        """Format schema for prompt"""
        
        if not schema:
            return "No schema provided"
        
        lines = []
        for table_name, columns in schema.items():
            if isinstance(columns, list):
                lines.append(f"{table_name}({', '.join(columns)})")
        
        return "\n".join(lines)
    
    def select_best_candidate(
        self,
        candidates: List[Candidate],
        evaluator: Callable[[str], Tuple[bool, bool, float]]
    ) -> Candidate:
        """
        Select best candidate through evaluation
        
        Args:
            candidates: List of SQL candidates
            evaluator: Function (sql -> (exact_match, exec_match, score))
            
        Returns:
            Best candidate with score
        """
        logger.info(f"Evaluating {len(candidates)} candidates")
        
        for candidate in candidates:
            try:
                exact_match, exec_match, score = evaluator(candidate.sql)
                
                # Calculate combined score
                candidate.score = score
                
                # Calculate confidence based on generation method
                base_confidence = 0.7
                if candidate.strategy_used and candidate.strategy_used != 'none':
                    base_confidence = 0.85
                
                # Adjust by temperature (lower = more confident)
                temp_factor = 1.0 - (candidate.temperature * 0.3)
                candidate.confidence = base_confidence * temp_factor
                
            except Exception as e:
                logger.error(f"Failed to evaluate candidate: {e}")
                candidate.score = 0.0
                candidate.confidence = 0.0
        
        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        best = candidates[0]
        logger.info(f"Best candidate: score={best.score:.2f}, confidence={best.confidence:.2f}")
        
        return best
    
    def _generate_id(self, prefix: str = "cand") -> str:
        """Generate unique candidate ID"""
        timestamp = str(time.time())
        hash_input = f"{prefix}_{timestamp}".encode()
        return f"{prefix}_{hashlib.md5(hash_input).hexdigest()[:8]}"


class SequentialScaling:
    """
    Sequential test-time scaling: Iterative refinement with feedback
    
    Features:
    - Memory-guided initial generation
    - Self-evaluation and feedback
    - Iterative refinement
    - Error detection and correction
    - Progressive improvement
    """
    
    def __init__(
        self,
        memory_store: ReasoningMemoryStore,
        memory_retrieval: MemoryRetrieval,
        sql_generator: Callable,
        sql_evaluator: Optional[Callable] = None
    ):
        """
        Initialize sequential scaling
        
        Args:
            memory_store: ReasoningMemoryStore instance
            memory_retrieval: MemoryRetrieval instance
            sql_generator: Function to generate SQL
            sql_evaluator: Function to evaluate SQL (optional)
        """
        self.memory_store = memory_store
        self.memory_retrieval = memory_retrieval
        self.sql_generator = sql_generator
        self.sql_evaluator = sql_evaluator
        
        logger.info("SequentialScaling initialized")
    
    def iterative_refinement(
        self,
        context: RetrievalContext,
        max_iterations: int = 3,
        enable_memory_guidance: bool = True
    ) -> ScalingResult:
        """
        Generate SQL through iterative refinement
        
        Args:
            context: Retrieval context
            max_iterations: Maximum refinement iterations
            enable_memory_guidance: Use retrieved strategies
            
        Returns:
            ScalingResult with best candidate and history
        """
        logger.info(f"Starting iterative refinement (max {max_iterations} iterations)")
        start_time = time.time()
        
        # Retrieve strategies
        strategies = []
        if enable_memory_guidance:
            retrieval_results = self.memory_retrieval.retrieve_strategies(
                context=context,
                top_k=3,
                min_confidence=0.5
            )
            strategies = [r.strategy for r in retrieval_results]
        
        # Initialize
        current_sql = None
        all_candidates = []
        best_score = 0.0
        best_candidate = None
        
        for iteration in range(max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{max_iterations}")
            
            if iteration == 0:
                # Initial generation
                candidate = self._generate_initial(context, strategies)
            else:
                # Refinement based on feedback
                feedback = self._generate_feedback(
                    current_sql,
                    all_candidates[-1]
                )
                candidate = self._generate_refinement(
                    context,
                    current_sql,
                    feedback,
                    strategies
                )
            
            # Evaluate if evaluator available
            if self.sql_evaluator:
                try:
                    exact_match, exec_match, score = self.sql_evaluator(candidate.sql)
                    candidate.score = score
                    
                    # Track best
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                    
                    # Early stopping if perfect
                    if exact_match and exec_match:
                        logger.info(f"Perfect match found at iteration {iteration + 1}")
                        break
                        
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    candidate.score = 0.0
            
            all_candidates.append(candidate)
            current_sql = candidate.sql
        
        total_time = time.time() - start_time
        
        # Use best if evaluator available, otherwise use last
        if best_candidate is None:
            best_candidate = all_candidates[-1]
        
        result = ScalingResult(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            total_time=total_time,
            num_candidates=len(all_candidates),
            scaling_method='sequential',
            improvement_metrics={
                'iterations': len(all_candidates),
                'initial_score': all_candidates[0].score if all_candidates else 0.0,
                'final_score': best_candidate.score,
                'improvement': best_candidate.score - (all_candidates[0].score if all_candidates else 0.0)
            }
        )
        
        logger.info(f"Refinement complete: {len(all_candidates)} iterations in {total_time:.2f}s")
        return result
    
    def _generate_initial(
        self,
        context: RetrievalContext,
        strategies: List[ReasoningStrategy]
    ) -> Candidate:
        """Generate initial SQL"""
        
        start_time = time.time()
        
        # Build initial prompt
        prompt = self._build_initial_prompt(context, strategies)
        
        # Generate
        sql = self.sql_generator(prompt=prompt, temperature=0.1)
        
        generation_time = time.time() - start_time
        
        return Candidate(
            candidate_id=self._generate_id("seq_0"),
            sql=sql,
            strategy_used=strategies[0].name if strategies else 'none',
            generation_method='sequential_initial',
            temperature=0.1,
            generation_time=generation_time,
            confidence=0.8
        )
    
    def _generate_refinement(
        self,
        context: RetrievalContext,
        current_sql: str,
        feedback: str,
        strategies: List[ReasoningStrategy]
    ) -> Candidate:
        """Generate refinement based on feedback"""
        
        start_time = time.time()
        
        # Build refinement prompt
        prompt = self._build_refinement_prompt(
            context,
            current_sql,
            feedback,
            strategies
        )
        
        # Generate with slightly higher temperature for exploration
        sql = self.sql_generator(prompt=prompt, temperature=0.2)
        
        generation_time = time.time() - start_time
        
        return Candidate(
            candidate_id=self._generate_id("seq_ref"),
            sql=sql,
            strategy_used=strategies[0].name if strategies else 'none',
            generation_method='sequential_refinement',
            temperature=0.2,
            generation_time=generation_time,
            confidence=0.75,
            metadata={'feedback': feedback}
        )
    
    def _generate_feedback(
        self,
        sql: str,
        candidate: Candidate
    ) -> str:
        """Generate feedback for refinement"""
        
        feedback_parts = []
        
        # Check for common issues
        sql_upper = sql.upper()
        
        # Issue 1: Missing GROUP BY
        if any(agg in sql_upper for agg in ['AVG', 'SUM', 'COUNT', 'MAX', 'MIN']):
            if 'GROUP BY' not in sql_upper:
                feedback_parts.append("Consider adding GROUP BY if aggregating per group")
        
        # Issue 2: Ambiguous columns
        if 'JOIN' in sql_upper and '.' not in sql:
            feedback_parts.append("Use table prefixes (aliases) to avoid ambiguous columns")
        
        # Issue 3: Ordering
        if 'ORDER BY' in sql_upper and 'GROUP BY' in sql_upper:
            order_idx = sql_upper.index('ORDER BY')
            group_idx = sql_upper.index('GROUP BY')
            if order_idx < group_idx:
                feedback_parts.append("ORDER BY should come after GROUP BY")
        
        # Issue 4: Check evaluation score
        if candidate.score < 0.5:
            feedback_parts.append("Current SQL has low accuracy. Review the query structure.")
        
        if not feedback_parts:
            feedback_parts.append("SQL looks good, but try to optimize further")
        
        return " | ".join(feedback_parts)
    
    def _build_initial_prompt(
        self,
        context: RetrievalContext,
        strategies: List[ReasoningStrategy]
    ) -> str:
        """Build initial generation prompt"""
        
        prompt = f"""
Database Schema:
{self._format_schema(context.schema)}

Question: {context.query}
"""
        
        if strategies:
            strategy = strategies[0]
            prompt += f"""

Recommended Strategy: {strategy.name}
Success Rate: {strategy.success_rate:.1%}

Key Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(strategy.reasoning_steps[:5]))}

Generate SQL query:"""
        else:
            prompt += "\nGenerate SQL query:"
        
        return prompt
    
    def _build_refinement_prompt(
        self,
        context: RetrievalContext,
        current_sql: str,
        feedback: str,
        strategies: List[ReasoningStrategy]
    ) -> str:
        """Build refinement prompt"""
        
        prompt = f"""
Database Schema:
{self._format_schema(context.schema)}

Question: {context.query}

Current SQL:
{current_sql}

Feedback:
{feedback}

Please refine the SQL query addressing the feedback.
"""
        
        if strategies:
            prompt += f"""
Strategy: {strategies[0].name}

Critical Rules:
{chr(10).join(f"- {rule}" for rule in strategies[0].critical_rules[:5])}
"""
        
        prompt += "\nRefined SQL query:"
        return prompt
    
    def _format_schema(self, schema: Dict) -> str:
        """Format schema for prompt"""
        
        if not schema:
            return "No schema provided"
        
        lines = []
        for table_name, columns in schema.items():
            if isinstance(columns, list):
                lines.append(f"{table_name}({', '.join(columns)})")
        
        return "\n".join(lines)
    
    def _generate_id(self, prefix: str = "seq") -> str:
        """Generate unique ID"""
        timestamp = str(time.time())
        hash_input = f"{prefix}_{timestamp}".encode()
        return f"{prefix}_{hashlib.md5(hash_input).hexdigest()[:8]}"


class HybridScaling:
    """
    Hybrid scaling: Combines parallel and sequential approaches
    
    Features:
    - Initial parallel generation for diversity
    - Sequential refinement of top candidates
    - Best of both worlds
    """
    
    def __init__(
        self,
        parallel_scaler: ParallelScaling,
        sequential_scaler: SequentialScaling
    ):
        """
        Initialize hybrid scaling
        
        Args:
            parallel_scaler: ParallelScaling instance
            sequential_scaler: SequentialScaling instance
        """
        self.parallel_scaler = parallel_scaler
        self.sequential_scaler = sequential_scaler
        
        logger.info("HybridScaling initialized")
    
    def generate_with_hybrid_scaling(
        self,
        context: RetrievalContext,
        n_parallel: int = 5,
        n_refine: int = 2,
        max_refinement_iterations: int = 2
    ) -> ScalingResult:
        """
        Generate SQL with hybrid approach
        
        Args:
            context: Retrieval context
            n_parallel: Number of parallel candidates
            n_refine: Number of top candidates to refine
            max_refinement_iterations: Max iterations per refinement
            
        Returns:
            ScalingResult with best candidate
        """
        logger.info("Starting hybrid scaling")
        start_time = time.time()
        
        # Phase 1: Parallel generation
        logger.info(f"Phase 1: Generating {n_parallel} parallel candidates")
        parallel_candidates = self.parallel_scaler.generate_diverse_candidates(
            context=context,
            n_candidates=n_parallel,
            enable_memory_guidance=True
        )
        
        # Select top candidates for refinement
        # (In practice, you'd evaluate them first)
        top_candidates = parallel_candidates[:n_refine]
        
        # Phase 2: Sequential refinement
        logger.info(f"Phase 2: Refining top {n_refine} candidates")
        refined_results = []
        
        for i, candidate in enumerate(top_candidates):
            logger.debug(f"Refining candidate {i+1}/{n_refine}")
            
            # Create context with current SQL as starting point
            # (This would use the sequential scaler with the candidate as initial)
            result = self.sequential_scaler.iterative_refinement(
                context=context,
                max_iterations=max_refinement_iterations,
                enable_memory_guidance=True
            )
            refined_results.append(result)
        
        # Select best across all refined candidates
        all_candidates = parallel_candidates + [
            c for r in refined_results for c in r.all_candidates
        ]
        
        # Find best (highest score)
        best_candidate = max(all_candidates, key=lambda c: c.score)
        
        total_time = time.time() - start_time
        
        result = ScalingResult(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            total_time=total_time,
            num_candidates=len(all_candidates),
            scaling_method='hybrid',
            improvement_metrics={
                'parallel_candidates': n_parallel,
                'refined_candidates': n_refine,
                'total_candidates': len(all_candidates),
                'best_score': best_candidate.score
            }
        )
        
        logger.info(f"Hybrid scaling complete: {len(all_candidates)} total candidates in {total_time:.2f}s")
        return result