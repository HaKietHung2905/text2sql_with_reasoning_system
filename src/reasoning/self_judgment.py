"""
Self-Judgment Module: Analyzes trajectories for success/failure patterns

This module automatically assesses whether each interaction trajectory
was successful or failed, and extracts insights for learning.

Based on ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re

from .experience_collector import Trajectory
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SuccessType(Enum):
    """Types of success classification"""
    COMPLETE_SUCCESS = "complete_success"      # Both exact match and execution match
    PARTIAL_SUCCESS = "partial_success"        # Execution match only (formatting issues)
    EXECUTION_FAILURE = "execution_failure"    # Logic errors, wrong results
    PARSE_FAILURE = "parse_failure"           # Syntax errors, unparseable SQL


@dataclass
class Insight:
    """Single insight extracted from trajectory analysis"""
    pattern: str                    # High-level pattern identified
    key_elements: List[str]         # Specific elements that contributed
    lesson: Optional[str] = None    # Lesson learned
    frequency: float = 1.0          # How common this pattern is
    confidence: float = 1.0         # Confidence in this insight


@dataclass
class JudgmentResult:
    """Complete result of trajectory judgment"""
    trajectory_id: str
    success_type: SuccessType
    confidence: float
    insights: List[Insight]
    key_factors: Dict[str, any]
    
    # Additional metrics
    component_analysis: Optional[Dict[str, float]] = None
    error_categories: Optional[List[str]] = None
    
    def is_success(self) -> bool:
        """Check if trajectory was successful"""
        return self.success_type in [
            SuccessType.COMPLETE_SUCCESS,
            SuccessType.PARTIAL_SUCCESS
        ]
    
    def is_complete_success(self) -> bool:
        """Check if trajectory was complete success"""
        return self.success_type == SuccessType.COMPLETE_SUCCESS


class SelfJudgment:
    """
    Judges trajectory quality and extracts learning insights
    
    This module implements self-judgment capabilities:
    - Automatic success/failure classification
    - Insight extraction from both successes and failures
    - Key factor identification
    - Pattern recognition across trajectories
    """
    
    def __init__(self):
        """Initialize self-judgment module"""
        self.judgment_history = []
        self.pattern_frequencies = {}
        logger.info("SelfJudgment module initialized")
    
    def judge_trajectory(self, trajectory: Trajectory) -> JudgmentResult:
        """
        Judge trajectory and extract insights
        
        Args:
            trajectory: Trajectory to judge
            
        Returns:
            JudgmentResult with success type and insights
        """
        # Step 1: Determine success type and confidence
        success_type, confidence = self._determine_success_type(trajectory)
        
        # Step 2: Extract insights based on success type
        insights = self._extract_insights(trajectory, success_type)
        
        # Step 3: Identify key factors that influenced outcome
        key_factors = self._identify_key_factors(trajectory)
        
        # Step 4: Analyze component-level performance
        component_analysis = self._analyze_components(trajectory)
        
        # Step 5: Categorize errors if any
        error_categories = self._categorize_errors(trajectory)
        
        # Create judgment result
        result = JudgmentResult(
            trajectory_id=trajectory.trajectory_id,
            success_type=success_type,
            confidence=confidence,
            insights=insights,
            key_factors=key_factors,
            component_analysis=component_analysis,
            error_categories=error_categories
        )
        
        # Store in history
        self.judgment_history.append(result)
        
        logger.debug(f"Judged {trajectory.trajectory_id}: {success_type.value} (confidence: {confidence:.2f})")
        return result
    
    def _determine_success_type(
        self,
        trajectory: Trajectory
    ) -> Tuple[SuccessType, float]:
        """
        Determine success type and confidence level
        
        Returns:
            Tuple of (SuccessType, confidence_score)
        """
        
        # Priority 1: Check for parse/syntax errors
        if trajectory.errors and len(trajectory.errors) > 0:
            for error in trajectory.errors:
                error_lower = error.lower()
                if any(keyword in error_lower for keyword in ['parse', 'syntax', 'invalid', 'malformed']):
                    return SuccessType.PARSE_FAILURE, 1.0
        
        # Priority 2: Check if SQL was generated
        if not trajectory.generated_sql or len(trajectory.generated_sql.strip()) == 0:
            return SuccessType.PARSE_FAILURE, 1.0
        
        # Priority 3: Check evaluation results
        if trajectory.exact_match and trajectory.execution_match:
            # Perfect match - highest confidence
            return SuccessType.COMPLETE_SUCCESS, 1.0
        
        elif trajectory.execution_match and not trajectory.exact_match:
            # Works but formatting issues
            # Confidence based on how close it was
            confidence = self._calculate_partial_success_confidence(trajectory)
            return SuccessType.PARTIAL_SUCCESS, confidence
        
        elif not trajectory.execution_match:
            # Logic errors - results don't match
            return SuccessType.EXECUTION_FAILURE, 1.0
        
        else:
            # Fallback - unclear case
            return SuccessType.EXECUTION_FAILURE, 0.5
    
    def _calculate_partial_success_confidence(self, trajectory: Trajectory) -> float:
        """Calculate confidence for partial success cases"""
        
        confidence = 0.7  # Base confidence
        
        # Adjust based on component scores
        if trajectory.component_scores:
            avg_f1 = self._calculate_average_f1(trajectory.component_scores)
            # Higher component scores = higher confidence
            confidence = 0.5 + (avg_f1 * 0.4)  # Range: 0.5 to 0.9
        
        return confidence
    
    def _calculate_average_f1(self, component_scores: Dict) -> float:
        """Calculate average F1 score across components"""
        
        f1_scores = []
        
        for component, scores in component_scores.items():
            if isinstance(scores, dict) and 'f1' in scores:
                f1_scores.append(scores['f1'])
        
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    def _extract_insights(
        self,
        trajectory: Trajectory,
        success_type: SuccessType
    ) -> List[Insight]:
        """
        Extract insights from trajectory based on success type
        
        Args:
            trajectory: Trajectory to analyze
            success_type: Type of success/failure
            
        Returns:
            List of insights
        """
        
        insights = []
        
        if success_type == SuccessType.COMPLETE_SUCCESS:
            insights.extend(self._extract_success_insights(trajectory))
        
        elif success_type == SuccessType.PARTIAL_SUCCESS:
            insights.extend(self._extract_partial_success_insights(trajectory))
        
        elif success_type == SuccessType.EXECUTION_FAILURE:
            insights.extend(self._extract_failure_insights(trajectory))
        
        elif success_type == SuccessType.PARSE_FAILURE:
            insights.extend(self._extract_parse_failure_insights(trajectory))
        
        return insights
    
    def _extract_success_insights(self, trajectory: Trajectory) -> List[Insight]:
        """Extract insights from successful trajectories"""
        
        insights = []
        sql_upper = trajectory.generated_sql.upper()
        
        # Insight 1: Analyze query type
        query_type = self._identify_query_type(trajectory.generated_sql)
        if query_type:
            insights.append(Insight(
                pattern=f"Successful {query_type} query",
                key_elements=[
                    f"Query type: {query_type}",
                    f"Database: {trajectory.database}",
                    f"Execution time: {trajectory.execution_time:.3f}s" if trajectory.execution_time else "Execution time: N/A"
                ],
                lesson=f"Effective pattern for {query_type} queries",
                confidence=1.0
            ))
        
        # Insight 2: Semantic analysis effectiveness
        if trajectory.semantic_analysis:
            intent = trajectory.semantic_analysis.get('intent', '')
            if intent:
                insights.append(Insight(
                    pattern="Semantic analysis guided success",
                    key_elements=[
                        f"Detected intent: {intent}",
                        "Intent correctly mapped to SQL structure"
                    ],
                    lesson="Semantic layer provides valuable guidance",
                    confidence=0.9
                ))
        
        # Insight 3: Strategy retrieval effectiveness
        if trajectory.retrieved_strategies and len(trajectory.retrieved_strategies) > 0:
            top_strategy = trajectory.retrieved_strategies[0]
            insights.append(Insight(
                pattern="Strategy-guided success",
                key_elements=[
                    f"Used strategy: {top_strategy.get('name', 'unknown')}",
                    f"Strategy relevance: {top_strategy.get('relevance_score', 0):.2%}",
                    f"Strategy success rate: {top_strategy.get('success_rate', 0):.2%}"
                ],
                lesson="Retrieved strategy was highly effective",
                confidence=0.95
            ))
        
        # Insight 4: Example retrieval effectiveness
        retrieved_examples = getattr(trajectory, 'retrieved_examples', None)
        if retrieved_examples and len(retrieved_examples) > 0:
            top_example = retrieved_examples[0]
            similarity = top_example.get('similarity', 0)
            if similarity > 0.8:
                insights.append(Insight(
                    pattern="High-quality example retrieval",
                    key_elements=[
                        f"Top example similarity: {similarity:.2%}",
                        f"Retrieved {len(trajectory.retrieved_examples)} examples"
                    ],
                    lesson="Similar examples provide strong guidance",
                    confidence=0.85
                ))
        
        # Insight 5: Complex query success patterns
        if 'JOIN' in sql_upper and 'GROUP BY' in sql_upper:
            insights.append(Insight(
                pattern="Successful complex multi-table aggregation",
                key_elements=[
                    "Correctly handled table joins",
                    "Proper GROUP BY usage",
                    "Aggregation functions applied correctly"
                ],
                lesson="System handles complex multi-step queries well",
                confidence=0.9
            ))
        
        # Insight 6: Fast generation
        if trajectory.generation_time < 1.0:
            insights.append(Insight(
                pattern="Fast successful generation",
                key_elements=[
                    f"Generation time: {trajectory.generation_time:.3f}s",
                    "Efficient processing"
                ],
                lesson="Quick generation doesn't compromise quality",
                confidence=0.8
            ))
        
        return insights
    
    def _extract_partial_success_insights(
        self,
        trajectory: Trajectory
    ) -> List[Insight]:
        """Extract insights from partial success (execution match but not exact match)"""
        
        insights = []
        
        # Main insight: Logic is correct but formatting differs
        insights.append(Insight(
            pattern="Correct logic with formatting mismatch",
            key_elements=[
                "Query executed successfully",
                "Results match gold standard",
                "Structural differences in SQL syntax"
            ],
            lesson="Focus on Spider format compliance in post-processing",
            confidence=0.9
        ))
        
        # Analyze specific formatting issues
        if trajectory.component_scores:
            formatting_issues = []
            
            for component, scores in trajectory.component_scores.items():
                if isinstance(scores, dict):
                    f1 = scores.get('f1', 1.0)
                    
                    # Component with low score indicates formatting issue
                    if f1 < 0.8:
                        formatting_issues.append(f"{component} (F1: {f1:.2f})")
            
            if formatting_issues:
                insights.append(Insight(
                    pattern="Specific component formatting issues",
                    key_elements=formatting_issues,
                    lesson=f"Pay special attention to: {', '.join(formatting_issues)}",
                    confidence=0.85
                ))
        
        # Check for common formatting differences
        sql_lower = trajectory.generated_sql.lower()
        
        # Case sensitivity issues
        if any(func in trajectory.generated_sql for func in ['avg', 'sum', 'count', 'max', 'min']):
            insights.append(Insight(
                pattern="Function case sensitivity issue",
                key_elements=[
                    "Aggregation functions in lowercase",
                    "Spider expects uppercase: AVG, SUM, COUNT, etc."
                ],
                lesson="Always use uppercase for SQL functions",
                confidence=0.8
            ))
        
        # Backtick usage
        if '`' in trajectory.generated_sql:
            insights.append(Insight(
                pattern="Backtick usage in SQL",
                key_elements=[
                    "SQL contains backticks",
                    "Spider format doesn't use backticks"
                ],
                lesson="Remove backticks in post-processing",
                confidence=0.85
            ))
        
        return insights
    
    def _extract_failure_insights(self, trajectory: Trajectory) -> List[Insight]:
        """Extract insights from execution failures"""
        
        insights = []
        sql_upper = trajectory.generated_sql.upper()
        
        # Insight 1: Component-level failures
        if trajectory.component_scores:
            failed_components = []
            
            for component, scores in trajectory.component_scores.items():
                if isinstance(scores, dict):
                    f1 = scores.get('f1', 1.0)
                    precision = scores.get('precision', 1.0)
                    recall = scores.get('recall', 1.0)
                    
                    if f1 < 0.5:  # Significant failure
                        failed_components.append({
                            'component': component,
                            'f1': f1,
                            'precision': precision,
                            'recall': recall
                        })
                        
                        insights.append(Insight(
                            pattern=f"Failed {component} construction",
                            key_elements=[
                                f"F1 score: {f1:.2f}",
                                f"Precision: {precision:.2f}",
                                f"Recall: {recall:.2f}"
                            ],
                            lesson=f"Need better strategy for {component} clauses",
                            confidence=0.9
                        ))
        
        # Insight 2: Missing critical SQL components
        query_lower = trajectory.question.lower()
        
        # Missing GROUP BY for aggregation
        if any(agg in sql_upper for agg in ['AVG', 'SUM', 'COUNT', 'MAX', 'MIN']):
            if 'GROUP BY' not in sql_upper and any(word in query_lower for word in ['each', 'every', 'per', 'by']):
                insights.append(Insight(
                    pattern="Missing GROUP BY for aggregation",
                    key_elements=[
                        "Query contains aggregation function",
                        "Question implies grouping ('each', 'per', 'by')",
                        "No GROUP BY clause in generated SQL"
                    ],
                    lesson="Always add GROUP BY when aggregating per group",
                    confidence=0.95
                ))
        
        # Missing JOIN for multi-table query
        if 'FROM' in sql_upper:
            # Check if query mentions multiple entities
            if len(re.findall(r'\bFROM\b', sql_upper)) == 1 and 'JOIN' not in sql_upper:
                # Query might need join but doesn't have it
                if trajectory.schema and len(trajectory.schema.get('tables', [])) > 1:
                    insights.append(Insight(
                        pattern="Potentially missing table join",
                        key_elements=[
                            "Multi-table database",
                            "Single table in FROM clause",
                            "No JOIN clause present"
                        ],
                        lesson="Check if query requires data from multiple tables",
                        confidence=0.75
                    ))
        
        # Incorrect ORDER BY without aggregation alias
        if 'ORDER BY' in sql_upper and 'GROUP BY' in sql_upper:
            # Check if ordering by aggregated column
            order_by_match = re.search(r'ORDER BY\s+(\w+)', sql_upper)
            if order_by_match:
                order_col = order_by_match.group(1)
                if any(agg in sql_upper.split('SELECT')[1].split('FROM')[0] for agg in ['AVG', 'SUM', 'COUNT']):
                    insights.append(Insight(
                        pattern="ORDER BY on aggregated column",
                        key_elements=[
                            "Query has GROUP BY and ORDER BY",
                            "Ordering by aggregated result",
                            "Should use alias or full aggregation expression"
                        ],
                        lesson="Use alias for aggregated columns in ORDER BY",
                        confidence=0.8
                    ))
        
        # Insight 3: Schema understanding issues
        if trajectory.semantic_analysis:
            metrics = trajectory.semantic_analysis.get('metrics', [])
            dimensions = trajectory.semantic_analysis.get('dimensions', [])
            
            if not metrics and 'aggregation' in trajectory.semantic_analysis.get('intent', '').lower():
                insights.append(Insight(
                    pattern="Failed to identify aggregation metric",
                    key_elements=[
                        "Intent suggests aggregation",
                        "No metrics identified",
                        "Likely incorrect column selection"
                    ],
                    lesson="Improve metric identification in semantic layer",
                    confidence=0.85
                ))
        
        # Insight 4: Strategy mismatch
        if trajectory.retrieved_strategies:
            # Check if retrieved strategies were appropriate
            query_type = self._identify_query_type(trajectory.generated_sql)
            strategy_patterns = [s.get('pattern', '') for s in trajectory.retrieved_strategies]
            
            if query_type and query_type not in ' '.join(strategy_patterns):
                insights.append(Insight(
                    pattern="Strategy-query type mismatch",
                    key_elements=[
                        f"Generated query type: {query_type}",
                        f"Retrieved strategies: {', '.join(strategy_patterns[:2])}",
                        "Mismatch between strategy and actual query structure"
                    ],
                    lesson="Improve strategy retrieval relevance",
                    confidence=0.7
                ))
        
        return insights
    
    def _extract_parse_failure_insights(
        self,
        trajectory: Trajectory
    ) -> List[Insight]:
        """Extract insights from parse failures"""
        
        insights = []
        
        # Analyze errors
        if trajectory.errors:
            for error in trajectory.errors:
                error_lower = error.lower()
                
                # Syntax errors
                if 'syntax' in error_lower:
                    insights.append(Insight(
                        pattern="SQL syntax error",
                        key_elements=[f"Error: {error}"],
                        lesson="Review prompt instructions for SQL syntax rules",
                        confidence=1.0
                    ))
                
                # Missing keywords
                if 'missing' in error_lower or 'expected' in error_lower:
                    insights.append(Insight(
                        pattern="Missing SQL keyword or clause",
                        key_elements=[f"Error: {error}"],
                        lesson="Ensure all required SQL keywords are present",
                        confidence=0.9
                    ))
                
                # Invalid identifiers
                if 'invalid' in error_lower or 'unknown' in error_lower:
                    insights.append(Insight(
                        pattern="Invalid column or table name",
                        key_elements=[f"Error: {error}"],
                        lesson="Validate column/table names against schema",
                        confidence=0.95
                    ))
        
        # General parse failure insight
        if not trajectory.errors or len(trajectory.errors) == 0:
            insights.append(Insight(
                pattern="Parse failure without specific error",
                key_elements=[
                    "SQL could not be parsed",
                    "No specific error message available"
                ],
                lesson="Add more detailed error logging",
                confidence=0.6
            ))
        
        return insights
    
    def _identify_key_factors(self, trajectory: Trajectory) -> Dict:
        """
        Identify key factors that influenced the outcome
        
        Returns:
            Dictionary of key factors with values
        """
        
        sql_upper = trajectory.generated_sql.upper()
        
        factors = {
            # Input characteristics
            'query_difficulty': getattr(trajectory, 'difficulty', None),
            'query_length': len(trajectory.question.split()),
            'database': trajectory.database,
            'num_tables_in_schema': len(trajectory.schema.get('tables', [])) if trajectory.schema else 0,
            
            # Processing characteristics
            'has_semantic_analysis': trajectory.semantic_analysis is not None,
            'num_retrieved_examples': len(getattr(trajectory, 'retrieved_examples', None) or []),
            'num_retrieved_strategies': len(getattr(trajectory, 'retrieved_strategies', None) or []),
            'prompt_length': len(getattr(trajectory, 'prompt_used', None) or ''),
            
            # Output characteristics
            'generation_time': trajectory.generation_time,
            'sql_length': len(trajectory.generated_sql),
            'sql_word_count': len(trajectory.generated_sql.split()),
            
            # SQL structure characteristics
            'has_join': 'JOIN' in sql_upper,
            'num_joins': sql_upper.count('JOIN'),
            'has_aggregation': any(agg in sql_upper for agg in ['AVG', 'SUM', 'COUNT', 'MAX', 'MIN']),
            'has_group_by': 'GROUP BY' in sql_upper,
            'has_order_by': 'ORDER BY' in sql_upper,
            'has_limit': 'LIMIT' in sql_upper,
            'has_where': 'WHERE' in sql_upper,
            'has_having': 'HAVING' in sql_upper,
            'has_subquery': sql_upper.count('SELECT') > 1,
            'has_distinct': 'DISTINCT' in sql_upper,
            
            # Execution characteristics
            'execution_time': getattr(trajectory, 'execution_time', None) or 0.0,
            'has_errors': bool(getattr(trajectory, 'errors', None) and len(trajectory.errors) > 0),
            'num_errors': len(getattr(trajectory, 'errors', None) or []),
        }
        
        return factors
    
    def _analyze_components(self, trajectory: Trajectory) -> Optional[Dict[str, float]]:
        """Analyze component-level performance"""
        
        if not trajectory.component_scores:
            return None
        
        analysis = {}
        
        for component, scores in trajectory.component_scores.items():
            if isinstance(scores, dict):
                f1 = scores.get('f1', 0.0)
                analysis[component] = f1
        
        return analysis
    
    def _categorize_errors(self, trajectory: Trajectory) -> Optional[List[str]]:
        """Categorize errors into types"""
        
        if not trajectory.errors or len(trajectory.errors) == 0:
            return None
        
        categories = set()
        
        for error in trajectory.errors:
            error_lower = error.lower()
            
            if 'syntax' in error_lower:
                categories.add('SYNTAX_ERROR')
            elif 'parse' in error_lower:
                categories.add('PARSE_ERROR')
            elif 'execution' in error_lower or 'runtime' in error_lower:
                categories.add('EXECUTION_ERROR')
            elif 'column' in error_lower or 'table' in error_lower:
                categories.add('SCHEMA_ERROR')
            elif 'timeout' in error_lower:
                categories.add('TIMEOUT_ERROR')
            else:
                categories.add('UNKNOWN_ERROR')
        
        return list(categories)
    
    def _identify_query_type(self, sql: str) -> Optional[str]:
        """Identify the type of SQL query"""
        
        sql_upper = sql.upper()
        
        # Complex query patterns
        if 'JOIN' in sql_upper and 'GROUP BY' in sql_upper and 'ORDER BY' in sql_upper:
            return "MULTI_TABLE_AGGREGATION_RANKING"
        
        if 'JOIN' in sql_upper and 'GROUP BY' in sql_upper:
            return "MULTI_TABLE_AGGREGATION"
        
        if 'GROUP BY' in sql_upper and 'ORDER BY' in sql_upper and 'LIMIT' in sql_upper:
            return "AGGREGATION_RANKING"
        
        if 'JOIN' in sql_upper:
            return "MULTI_TABLE_JOIN"
        
        if 'GROUP BY' in sql_upper:
            return "AGGREGATION_GROUPING"
        
        if sql_upper.count('SELECT') > 1:
            return "NESTED_SUBQUERY"
        
        if 'WHERE' in sql_upper:
            return "FILTERED_SELECT"
        
        return "BASIC_SELECT"
    
    def batch_judge(self, trajectories: List[Trajectory]) -> List[JudgmentResult]:
        """
        Judge multiple trajectories in batch
        
        Args:
            trajectories: List of trajectories to judge
            
        Returns:
            List of judgment results
        """
        
        results = []
        
        for trajectory in trajectories:
            try:
                result = self.judge_trajectory(trajectory)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to judge trajectory {trajectory.trajectory_id}: {e}")
        
        logger.info(f"Batch judged {len(results)}/{len(trajectories)} trajectories")
        
        # Update pattern frequencies
        self._update_pattern_frequencies(results)
        
        return results
    
    def _update_pattern_frequencies(self, results: List[JudgmentResult]):
        """Update frequency counts for patterns"""
        
        for result in results:
            for insight in result.insights:
                pattern = insight.pattern
                if pattern not in self.pattern_frequencies:
                    self.pattern_frequencies[pattern] = 0
                self.pattern_frequencies[pattern] += 1
    
    def get_statistics(self) -> Dict:
        """Get judgment statistics"""
        
        if not self.judgment_history:
            return {}
        
        total = len(self.judgment_history)
        
        success_counts = {
            SuccessType.COMPLETE_SUCCESS: 0,
            SuccessType.PARTIAL_SUCCESS: 0,
            SuccessType.EXECUTION_FAILURE: 0,
            SuccessType.PARSE_FAILURE: 0
        }
        
        for result in self.judgment_history:
            success_counts[result.success_type] += 1
        
        return {
            'total_judgments': total,
            'complete_success': success_counts[SuccessType.COMPLETE_SUCCESS],
            'partial_success': success_counts[SuccessType.PARTIAL_SUCCESS],
            'execution_failure': success_counts[SuccessType.EXECUTION_FAILURE],
            'parse_failure': success_counts[SuccessType.PARSE_FAILURE],
            'complete_success_rate': success_counts[SuccessType.COMPLETE_SUCCESS] / total,
            'overall_success_rate': (success_counts[SuccessType.COMPLETE_SUCCESS] + 
                                    success_counts[SuccessType.PARTIAL_SUCCESS]) / total,
            'top_patterns': sorted(
                self.pattern_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def get_insights_summary(self) -> Dict[str, List[Insight]]:
        """Get summary of all insights by category"""
        
        summary = {
            'success_patterns': [],
            'failure_patterns': [],
            'improvement_opportunities': []
        }
        
        for result in self.judgment_history:
            for insight in result.insights:
                if result.is_complete_success():
                    summary['success_patterns'].append(insight)
                elif not result.is_success():
                    summary['failure_patterns'].append(insight)
                
                if insight.lesson:
                    summary['improvement_opportunities'].append(insight)
        
        return summary