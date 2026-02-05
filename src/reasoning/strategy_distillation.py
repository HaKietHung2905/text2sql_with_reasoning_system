"""
Strategy Distillation: Extracts generalizable reasoning strategies
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import defaultdict, Counter

from .experience_collector import Trajectory
from .self_judgment import JudgmentResult, SuccessType, Insight
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningStrategy:
    """High-level reasoning strategy"""
    
    # Identifiers
    strategy_id: str
    name: str
    pattern: str
    description: str
    
    # Applicability
    applicability: Dict[str, List[str]]
    
    # Strategy content
    reasoning_steps: List[str]
    critical_rules: List[str]
    sql_template_hints: Dict[str, str]
    common_pitfalls: List[Dict[str, str]] = field(default_factory=list)
    
    # Performance
    success_rate: float = 0.0
    sample_count: int = 0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    
    # For retrieval
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class StrategyDistillation:
    """Distills high-level strategies from trajectories"""
    
    def __init__(self, min_samples: int = 3):
        """
        Initialize strategy distillation
        
        Args:
            min_samples: Minimum successful examples to create strategy
        """
        self.min_samples = min_samples
        logger.info(f"StrategyDistillation initialized (min_samples={min_samples})")
    
    def distill_strategies(
        self,
        trajectories: List[Trajectory],
        judgments: List[JudgmentResult]
    ) -> List[ReasoningStrategy]:
        """
        Distill strategies from judged trajectories
        
        Args:
            trajectories: List of trajectories
            judgments: List of judgment results
            
        Returns:
            List of distilled strategies
        """
        # Group trajectories by pattern
        grouped = self._group_by_pattern(trajectories, judgments)
        
        strategies = []
        
        for pattern, group_data in grouped.items():
            if len(group_data['successful']) < self.min_samples:
                logger.debug(f"Skipping {pattern}: insufficient samples ({len(group_data['successful'])})")
                continue
            
            # Extract strategy from group
            strategy = self._extract_strategy(pattern, group_data)
            
            if strategy:
                strategies.append(strategy)
                logger.info(f"Distilled strategy: {strategy.name}")
        
        return strategies
    
    def _group_by_pattern(
        self,
        trajectories: List[Trajectory],
        judgments: List[JudgmentResult]
    ) -> Dict[str, Dict]:
        """Group trajectories by common patterns"""
        
        grouped = defaultdict(lambda: {
            'successful': [],
            'failed': [],
            'partial': []
        })
        
        for traj, judgment in zip(trajectories, judgments):
            # Identify pattern
            pattern = self._identify_pattern(traj)
            
            # Categorize by success type
            if judgment.success_type == SuccessType.COMPLETE_SUCCESS:
                grouped[pattern]['successful'].append((traj, judgment))
            elif judgment.success_type == SuccessType.PARTIAL_SUCCESS:
                grouped[pattern]['partial'].append((traj, judgment))
            else:
                grouped[pattern]['failed'].append((traj, judgment))
        
        return dict(grouped)
    
    def _identify_pattern(self, trajectory: Trajectory) -> str:
        """Identify the pattern/type of query"""
        
        sql = trajectory.generated_sql.upper()
        patterns = []
        
        # Check for aggregation
        if any(agg in sql for agg in ['AVG', 'SUM', 'COUNT', 'MAX', 'MIN']):
            patterns.append('AGGREGATION')
        
        # Check for grouping
        if 'GROUP BY' in sql:
            patterns.append('GROUPING')
        
        # Check for ordering
        if 'ORDER BY' in sql:
            patterns.append('ORDERING')
        
        # Check for limit
        if 'LIMIT' in sql:
            patterns.append('RANKING')
        
        # Check for joins
        if 'JOIN' in sql:
            patterns.append('JOIN')
        
        # Check for subqueries
        if sql.count('SELECT') > 1:
            patterns.append('SUBQUERY')
        
        # Check for filtering
        if 'WHERE' in sql:
            patterns.append('FILTERING')
        
        # Combine patterns
        if not patterns:
            return 'BASIC_SELECT'
        
        return '_'.join(sorted(patterns))
    
    def _extract_strategy(
        self,
        pattern: str,
        group_data: Dict
    ) -> Optional[ReasoningStrategy]:
        """Extract strategy from grouped data"""
        
        successful = group_data['successful']
        failed = group_data['failed']
        
        if not successful:
            return None
        
        # Generate strategy name
        name = self._generate_strategy_name(pattern)
        
        # Extract common elements from successful cases
        reasoning_steps = self._extract_reasoning_steps(pattern, successful)
        critical_rules = self._extract_critical_rules(pattern, successful)
        sql_hints = self._extract_sql_hints(pattern, successful)
        pitfalls = self._extract_pitfalls(failed)
        
        # Calculate success rate
        total = len(successful) + len(failed)
        success_rate = len(successful) / total if total > 0 else 0
        
        # Determine applicability
        applicability = self._determine_applicability(pattern, successful)
        
        strategy = ReasoningStrategy(
            strategy_id=f"strat_{pattern.lower()}_{uuid.uuid4().hex[:8]}",
            name=name,
            pattern=pattern,
            description=f"Strategy for {pattern.replace('_', ' ').lower()} queries",
            applicability=applicability,
            reasoning_steps=reasoning_steps,
            critical_rules=critical_rules,
            sql_template_hints=sql_hints,
            common_pitfalls=pitfalls,
            success_rate=success_rate,
            sample_count=len(successful)
        )
        
        return strategy
    
    def _generate_strategy_name(self, pattern: str) -> str:
        """Generate human-readable strategy name"""
        
        # Map patterns to names
        name_map = {
            'AGGREGATION_GROUPING_ORDERING_RANKING': 'Aggregation with Top-N Ranking',
            'AGGREGATION_GROUPING': 'Grouped Aggregation',
            'AGGREGATION_FILTERING': 'Filtered Aggregation',
            'JOIN_FILTERING': 'Multi-Table Filtering with Joins',
            'JOIN_AGGREGATION_GROUPING': 'Cross-Table Aggregation',
            'SUBQUERY': 'Nested Subquery Pattern',
            'BASIC_SELECT': 'Simple Selection'
        }
        
        return name_map.get(pattern, pattern.replace('_', ' ').title())
    
    def _extract_reasoning_steps(
        self,
        pattern: str,
        successful: List[tuple]
    ) -> List[str]:
        """Extract reasoning steps for the pattern"""
        
        # Pattern-specific reasoning steps
        steps_map = {
            'AGGREGATION_GROUPING_ORDERING_RANKING': [
                "1. Identify the metric to aggregate (e.g., salary → AVG)",
                "2. Determine the grouping dimension (e.g., department)",
                "3. Check if tables need to be joined",
                "4. Construct SELECT with aggregation function",
                "5. Add GROUP BY clause for dimensions",
                "6. Add ORDER BY on aggregated column (DESC/ASC)",
                "7. Apply LIMIT for top-N results"
            ],
            'JOIN_FILTERING': [
                "1. Identify tables needed from schema",
                "2. Determine join conditions (foreign keys)",
                "3. Select columns from appropriate tables",
                "4. Construct JOIN clauses with ON conditions",
                "5. Add WHERE clause for filters",
                "6. Use table aliases for clarity"
            ]
        }
        
        return steps_map.get(pattern, [
            "1. Analyze query requirements",
            "2. Identify relevant tables and columns",
            "3. Construct SQL following standard patterns",
            "4. Apply filters and conditions",
            "5. Validate syntax"
        ])
    
    def _extract_critical_rules(
        self,
        pattern: str,
        successful: List[tuple]
    ) -> List[str]:
        """Extract critical rules"""
        
        rules = []
        
        if 'AGGREGATION' in pattern:
            rules.append("Always GROUP BY non-aggregated columns in SELECT")
            rules.append("Aggregation functions: AVG, SUM, COUNT, MAX, MIN must be uppercase")
        
        if 'GROUPING' in pattern:
            rules.append("GROUP BY must include all non-aggregated SELECT columns")
        
        if 'ORDERING' in pattern:
            rules.append("ORDER BY comes after GROUP BY and before LIMIT")
            rules.append("Use DESC for highest/most, ASC for lowest/least")
        
        if 'JOIN' in pattern:
            rules.append("Always use explicit JOIN with ON clause")
            rules.append("Use table aliases in multi-table queries")
            rules.append("Prefix all columns with table alias to avoid ambiguity")
        
        if 'RANKING' in pattern:
            rules.append("LIMIT must be a positive integer")
            rules.append("LIMIT comes after ORDER BY")
        
        return rules
    
    def _extract_sql_hints(
        self,
        pattern: str,
        successful: List[tuple]
    ) -> Dict[str, str]:
        """Extract SQL template hints"""
        
        hints = {}
        
        if pattern == 'AGGREGATION_GROUPING_ORDERING_RANKING':
            hints['structure'] = "SELECT [dim], [AGG]([metric]) AS [alias] FROM [table] [JOIN] GROUP BY [dim] ORDER BY [alias] DESC/ASC LIMIT [N]"
            hints['example'] = "SELECT d.name, AVG(e.salary) AS avg_sal FROM employees e JOIN departments d ON e.dept_id = d.id GROUP BY d.name ORDER BY avg_sal DESC LIMIT 3"
        
        elif pattern == 'JOIN_FILTERING':
            hints['structure'] = "SELECT [cols] FROM [table1] [alias1] JOIN [table2] [alias2