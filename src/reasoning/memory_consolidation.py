"""
Memory Consolidation: Strategy evolution and refinement

This module handles the continuous improvement of reasoning strategies
by analyzing new experiences and updating the memory bank.

Based on ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import copy

from .memory_store import ReasoningMemoryStore
from .strategy_distillation import ReasoningStrategy, StrategyDistillation
from .experience_collector import Trajectory
from .self_judgment import JudgmentResult, SuccessType
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConsolidationDecision:
    """Types of consolidation decisions"""
    KEEP = "keep"              # Strategy is performing well
    REFINE = "refine"          # Update with new insights
    SPLIT = "split"            # Split into specialized strategies
    MERGE = "merge"            # Merge with similar strategy
    DEPRECATE = "deprecate"    # Mark as low-performing
    DELETE = "delete"          # Remove completely


class MemoryConsolidation:
    """
    Memory consolidation system for strategy evolution
    
    Features:
    - Continuous strategy refinement
    - Performance-based strategy updates
    - Strategy splitting for specialization
    - Strategy merging for efficiency
    - Automatic deprecation of poor performers
    - Evolution tracking
    """
    
    def __init__(
        self,
        memory_store: ReasoningMemoryStore,
        distillation: StrategyDistillation
    ):
        """
        Initialize memory consolidation
        
        Args:
            memory_store: ReasoningMemoryStore instance
            distillation: StrategyDistillation instance
        """
        self.memory_store = memory_store
        self.distillation = distillation
        
        # Consolidation configuration
        self.config = {
            'min_applications': 10,           # Min applications to consider consolidation
            'performance_drop_threshold': 0.10,  # 10% drop triggers investigation
            'merge_similarity_threshold': 0.90,  # Similarity for merging
            'split_variance_threshold': 0.15,    # Performance variance for splitting
            'deprecation_threshold': 0.40,       # Success rate below this = deprecate
            'min_sample_count': 5,               # Min samples to keep strategy
            'refinement_frequency': 20           # Refine every N applications
        }
        
        # Track consolidation history
        self.consolidation_history = []
        
        logger.info("MemoryConsolidation initialized")
    
    def consolidate_memory(
        self,
        new_trajectories: List[Trajectory],
        new_judgments: List[JudgmentResult]
    ) -> Dict[str, int]:
        """
        Consolidate memory with new experiences
        
        Args:
            new_trajectories: List of new trajectories
            new_judgments: List of judgment results
            
        Returns:
            Dictionary with consolidation statistics
        """
        logger.info(f"Consolidating memory with {len(new_trajectories)} new experiences")
        
        stats = {
            'kept': 0,
            'refined': 0,
            'split': 0,
            'merged': 0,
            'deprecated': 0,
            'deleted': 0,
            'new_created': 0
        }
        
        # Step 1: Get all existing strategies
        existing_strategies = self.memory_store.get_all_strategies(active_only=True)
        
        # Step 2: Match trajectories to strategies
        strategy_applications = self._match_trajectories_to_strategies(
            new_trajectories,
            new_judgments
        )
        
        # Step 3: Analyze each strategy's performance
        for strategy in existing_strategies:
            applications = strategy_applications.get(strategy.strategy_id, [])
            
            if not applications:
                # No new applications for this strategy
                stats['kept'] += 1
                continue
            
            # Decide what to do with this strategy
            decision = self._make_consolidation_decision(strategy, applications)
            
            if decision == ConsolidationDecision.REFINE:
                self._refine_strategy(strategy, applications)
                stats['refined'] += 1
            
            elif decision == ConsolidationDecision.SPLIT:
                new_strategies = self._split_strategy(strategy, applications)
                stats['split'] += 1
                stats['new_created'] += len(new_strategies)
            
            elif decision == ConsolidationDecision.DEPRECATE:
                self._deprecate_strategy(strategy, applications)
                stats['deprecated'] += 1
            
            elif decision == ConsolidationDecision.DELETE:
                self.memory_store.delete_strategy(strategy.strategy_id)
                stats['deleted'] += 1
            
            else:  # KEEP
                stats['kept'] += 1
        
        # Step 4: Look for merge opportunities
        merge_count = self._find_and_merge_similar_strategies()
        stats['merged'] = merge_count
        
        # Step 5: Discover new patterns not covered by existing strategies
        new_strategies = self._discover_new_patterns(
            new_trajectories,
            new_judgments,
            existing_strategies
        )
        stats['new_created'] += len(new_strategies)
        
        logger.info(f"Consolidation complete: {stats}")
        return stats
        
    def _match_trajectories_to_strategies(
        self,
        trajectories: List[Trajectory],
        judgments: List[JudgmentResult]
    ) -> Dict[str, List[Tuple[Trajectory, JudgmentResult]]]:
        """
        Match trajectories to the strategies they used
        
        Args:
            trajectories: List of trajectories
            judgments: List of judgments
            
        Returns:
            Dict mapping strategy_id to list of (trajectory, judgment) tuples
        """
        strategy_applications = defaultdict(list)
        
        for traj, judgment in zip(trajectories, judgments):
            # Get strategies used in this trajectory
            # Handle different possible attribute names
            strategies_used = None
            
            if hasattr(traj, 'strategies_used') and traj.strategies_used:
                # strategies_used is a list of strategy IDs
                strategies_used = traj.strategies_used
            elif hasattr(traj, 'retrieved_strategies') and traj.retrieved_strategies:
                # retrieved_strategies might be a list of dicts
                if isinstance(traj.retrieved_strategies, list):
                    strategies_used = []
                    for item in traj.retrieved_strategies:
                        if isinstance(item, dict):
                            strategy_id = item.get('strategy_id')
                            if strategy_id:
                                strategies_used.append(strategy_id)
                        elif isinstance(item, str):
                            strategies_used.append(item)
            
            # Add to applications if we found strategies
            if strategies_used:
                for strategy_id in strategies_used:
                    strategy_applications[strategy_id].append((traj, judgment))
        
        logger.debug(f"Matched {len(strategy_applications)} strategies to trajectories")
        return dict(strategy_applications)

    
    def _make_consolidation_decision(
        self,
        strategy: ReasoningStrategy,
        applications: List[Tuple[Trajectory, JudgmentResult]]
    ) -> str:
        """Decide what to do with a strategy based on new applications"""
        
        if len(applications) < self.config['min_applications']:
            # Not enough data to make a decision
            return ConsolidationDecision.KEEP
        
        # Calculate new performance
        successes = sum(1 for _, j in applications if j.is_complete_success())
        new_success_rate = successes / len(applications)
        
        # Performance analysis
        performance_delta = new_success_rate - strategy.success_rate
        
        # Decision 1: Delete if consistently failing
        if (strategy.success_rate < self.config['deprecation_threshold'] and
            new_success_rate < self.config['deprecation_threshold'] and
            strategy.sample_count < self.config['min_sample_count']):
            return ConsolidationDecision.DELETE
        
        # Decision 2: Deprecate if significant performance drop
        if performance_delta < -self.config['performance_drop_threshold']:
            return ConsolidationDecision.DEPRECATE
        
        # Decision 3: Split if high variance in performance
        if self._should_split_strategy(strategy, applications):
            return ConsolidationDecision.SPLIT
        
        # Decision 4: Refine if performance is stable or improving
        if abs(performance_delta) > 0.02 or len(applications) >= self.config['refinement_frequency']:
            return ConsolidationDecision.REFINE
        
        # Default: Keep as is
        return ConsolidationDecision.KEEP
    
    def _refine_strategy(
        self,
        strategy: ReasoningStrategy,
        applications: List[Tuple[Trajectory, JudgmentResult]]
    ):
        """Refine strategy with new insights"""
        
        logger.info(f"Refining strategy: {strategy.name}")
        
        # Calculate new success rate (exponential moving average)
        new_successes = sum(1 for _, j in applications if j.is_complete_success())
        new_success_rate = new_successes / len(applications)
        
        old_success_rate = strategy.success_rate
        
        # EMA: 70% weight to old, 30% to new
        strategy.success_rate = 0.7 * old_success_rate + 0.3 * new_success_rate
        strategy.sample_count += len(applications)
        
        # Extract new lessons from failures
        failures = [(t, j) for t, j in applications if not j.is_success()]
        if failures:
            new_pitfalls = self._extract_new_pitfalls(failures)
            
            # Add new pitfalls, avoiding duplicates
            existing_patterns = {p['mistake'] for p in strategy.common_pitfalls}
            for pitfall in new_pitfalls:
                if pitfall['mistake'] not in existing_patterns:
                    strategy.common_pitfalls.append(pitfall)
            
            # Keep only top 10 most common pitfalls
            if len(strategy.common_pitfalls) > 10:
                # Sort by frequency
                strategy.common_pitfalls.sort(
                    key=lambda p: p.get('frequency', 0),
                    reverse=True
                )
                strategy.common_pitfalls = strategy.common_pitfalls[:10]
        
        # Extract new success patterns
        successes = [(t, j) for t, j in applications if j.is_complete_success()]
        if successes:
            self._enhance_with_success_patterns(strategy, successes)
        
        # Update metadata
        strategy.version += 1
        strategy.last_updated = datetime.now().isoformat()
        
        # Save updated strategy
        self.memory_store.store_strategy(strategy)
        
        # Log consolidation
        self.consolidation_history.append({
            'strategy_id': strategy.strategy_id,
            'action': 'refine',
            'old_success_rate': old_success_rate,
            'new_success_rate': strategy.success_rate,
            'new_applications': len(applications),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Refined {strategy.name}: {old_success_rate:.2%} → {strategy.success_rate:.2%}")
    
    def _extract_new_pitfalls(
        self,
        failures: List[Tuple[Trajectory, JudgmentResult]]
    ) -> List[Dict[str, any]]:
        """Extract new pitfalls from failures"""
        
        pitfalls = []
        error_patterns = Counter()
        
        for traj, judgment in failures:
            for insight in judgment.insights:
                if insight.lesson:
                    error_patterns[insight.pattern] += 1
        
        # Create pitfall entries
        total_failures = len(failures)
        for pattern, count in error_patterns.most_common(5):
            # Find an example
            for traj, judgment in failures:
                for insight in judgment.insights:
                    if insight.pattern == pattern:
                        pitfalls.append({
                            'mistake': pattern,
                            'consequence': insight.lesson or "Query failure",
                            'fix': insight.key_elements[0] if insight.key_elements else "Review query structure",
                            'frequency': count / total_failures
                        })
                        break
                if len(pitfalls) >= 5:
                    break
        
        return pitfalls
    
    def _enhance_with_success_patterns(
        self,
        strategy: ReasoningStrategy,
        successes: List[Tuple[Trajectory, JudgmentResult]]
    ):
        """Enhance strategy with patterns from successes"""
        
        # Extract common keywords from successful queries
        success_keywords = Counter()
        for traj, _ in successes:
            query_words = traj.query.lower().split()
            success_keywords.update(query_words)
        
        # Add top keywords to applicability
        existing_keywords = set(strategy.applicability.get('keywords', []))
        top_new_keywords = [
            word for word, count in success_keywords.most_common(10)
            if word not in existing_keywords and len(word) > 3
        ]
        
        if top_new_keywords:
            strategy.applicability['keywords'].extend(top_new_keywords[:5])
    
    def _should_split_strategy(
        self,
        strategy: ReasoningStrategy,
        applications: List[Tuple[Trajectory, JudgmentResult]]
    ) -> bool:
        """Determine if strategy should be split"""
        
        if len(applications) < 20:
            return False  # Need sufficient data
        
        # Analyze performance variance across different contexts
        performance_by_difficulty = defaultdict(list)
        performance_by_database = defaultdict(list)
        
        for traj, judgment in applications:
            success = 1.0 if judgment.is_complete_success() else 0.0
            
            if traj.difficulty:
                performance_by_difficulty[traj.difficulty].append(success)
            
            performance_by_database[traj.database].append(success)
        
        # Check variance
        max_variance = 0.0
        
        # Variance by difficulty
        if len(performance_by_difficulty) > 1:
            success_rates = [
                sum(scores) / len(scores)
                for scores in performance_by_difficulty.values()
                if len(scores) >= 3
            ]
            if len(success_rates) > 1:
                variance = max(success_rates) - min(success_rates)
                max_variance = max(max_variance, variance)
        
        # Variance by database
        if len(performance_by_database) > 1:
            success_rates = [
                sum(scores) / len(scores)
                for scores in performance_by_database.values()
                if len(scores) >= 3
            ]
            if len(success_rates) > 1:
                variance = max(success_rates) - min(success_rates)
                max_variance = max(max_variance, variance)
        
        # Split if variance is high
        return max_variance > self.config['split_variance_threshold']
    
    def _split_strategy(
        self,
        strategy: ReasoningStrategy,
        applications: List[Tuple[Trajectory, JudgmentResult]]
    ) -> List[ReasoningStrategy]:
        """Split strategy into specialized versions"""
        
        logger.info(f"Splitting strategy: {strategy.name}")
        
        # Group applications by context
        groups = self._group_applications_by_context(applications)
        
        new_strategies = []
        
        for context_key, group_applications in groups.items():
            if len(group_applications) < self.config['min_sample_count']:
                continue  # Skip small groups
            
            # Create specialized strategy
            specialized = copy.deepcopy(strategy)
            
            # Update metadata
            specialized.strategy_id = f"{strategy.strategy_id}_{context_key}"
            specialized.name = f"{strategy.name} ({context_key})"
            specialized.description = f"Specialized for {context_key}: {strategy.description}"
            specialized.version = 1
            specialized.created_at = datetime.now().isoformat()
            specialized.last_updated = datetime.now().isoformat()
            
            # Calculate performance for this context
            successes = sum(1 for _, j in group_applications if j.is_complete_success())
            specialized.success_rate = successes / len(group_applications)
            specialized.sample_count = len(group_applications)
            
            if specialized.applicability is None:
                specialized.applicability = {}
            specialized.applicability['context'] = context_key
            
            # Store specialized strategy
            self.memory_store.store_strategy(specialized)
            new_strategies.append(specialized)
        
        # Deprecate original strategy
        self._deprecate_strategy(strategy, applications)
        
        logger.info(f"Split {strategy.name} into {len(new_strategies)} specialized strategies")
        return new_strategies
    
    def _group_applications_by_context(
        self,
        applications: List[Tuple[Trajectory, JudgmentResult]]
    ) -> Dict[str, List]:
        """Group applications by context (difficulty, database, etc.)"""
        
        # Try grouping by difficulty first
        groups_by_difficulty = defaultdict(list)
        for traj, judgment in applications:
            # Safely get difficulty with fallback
            difficulty = None
            if hasattr(traj, 'difficulty') and traj.difficulty:
                difficulty = traj.difficulty
            elif hasattr(traj, 'metadata') and traj.metadata:
                difficulty = traj.metadata.get('difficulty')
            
            difficulty = difficulty or 'unknown'
            groups_by_difficulty[difficulty].append((traj, judgment))
        
        if len(groups_by_difficulty) > 1:
            valid_groups = {
                k: v for k, v in groups_by_difficulty.items()
                if len(v) >= self.config.get('min_sample_count', 5)
            }
            if len(valid_groups) > 1:
                return valid_groups
        
        # Fallback: group by database type
        groups_by_database = defaultdict(list)
        for traj, judgment in applications:
            database = getattr(traj, 'database', 'unknown')
            groups_by_database[database].append((traj, judgment))
        
        return dict(groups_by_database)
    
    def _deprecate_strategy(
        self,
        strategy: ReasoningStrategy,
        applications: List[Tuple[Trajectory, JudgmentResult]]
    ):
        """Deprecate a poorly performing strategy"""
        
        logger.info(f"Deprecating strategy: {strategy.name}")
        
        # Calculate final performance
        if applications:
            successes = sum(1 for _, j in applications if j.is_complete_success())
            final_success_rate = successes / len(applications)
        else:
            final_success_rate = strategy.success_rate
        
        # Deactivate in store
        self.memory_store.deactivate_strategy(strategy.strategy_id)
        
        # Log
        self.consolidation_history.append({
            'strategy_id': strategy.strategy_id,
            'action': 'deprecate',
            'final_success_rate': final_success_rate,
            'reason': 'Poor performance or superseded by specialized versions',
            'timestamp': datetime.now().isoformat()
        })
    
    def _find_and_merge_similar_strategies(self) -> int:
        """Find and merge highly similar strategies"""
        
        strategies = self.memory_store.get_all_strategies(active_only=True)
        
        if len(strategies) < 2:
            return 0
        
        merged_count = 0
        processed_ids = set()
        
        for i, strategy1 in enumerate(strategies):
            if strategy1.strategy_id in processed_ids:
                continue
            
            for strategy2 in strategies[i+1:]:
                if strategy2.strategy_id in processed_ids:
                    continue
                
                # Check similarity
                similarity = self._calculate_strategy_similarity(strategy1, strategy2)
                
                if similarity > self.config['merge_similarity_threshold']:
                    # Merge strategy2 into strategy1
                    self._merge_strategies(strategy1, strategy2)
                    processed_ids.add(strategy2.strategy_id)
                    merged_count += 1
        
        return merged_count
    
    def _calculate_strategy_similarity(
        self,
        strategy1: ReasoningStrategy,
        strategy2: ReasoningStrategy
    ) -> float:
        """Calculate similarity between two strategies"""
        
        # Same pattern = high similarity
        if strategy1.pattern == strategy2.pattern:
            pattern_score = 1.0
        else:
            # Check pattern overlap
            parts1 = set(strategy1.pattern.split('_'))
            parts2 = set(strategy2.pattern.split('_'))
            pattern_score = len(parts1 & parts2) / len(parts1 | parts2)
        
        # Keyword overlap
        keywords1 = set(strategy1.applicability.get('keywords', []))
        keywords2 = set(strategy2.applicability.get('keywords', []))
        
        if keywords1 and keywords2:
            keyword_score = len(keywords1 & keywords2) / len(keywords1 | keywords2)
        else:
            keyword_score = 0.0
        
        # Reasoning steps similarity (simple check)
        steps1 = set(strategy1.reasoning_steps)
        steps2 = set(strategy2.reasoning_steps)
        
        if steps1 and steps2:
            steps_score = len(steps1 & steps2) / len(steps1 | steps2)
        else:
            steps_score = 0.0
        
        # Weighted average
        similarity = (
            0.5 * pattern_score +
            0.3 * keyword_score +
            0.2 * steps_score
        )
        
        return similarity
    
    def _merge_strategies(
        self,
        primary: ReasoningStrategy,
        secondary: ReasoningStrategy
    ):
        """Merge secondary strategy into primary"""
        
        logger.info(f"Merging {secondary.name} into {primary.name}")
        
        # Combine sample counts
        total_samples = primary.sample_count + secondary.sample_count
        
        # Weighted average of success rates
        primary.success_rate = (
            (primary.success_rate * primary.sample_count +
             secondary.success_rate * secondary.sample_count) /
            total_samples
        )
        
        primary.sample_count = total_samples
        
        # Merge keywords (avoid duplicates)
        primary_keywords = set(primary.applicability.get('keywords', []))
        secondary_keywords = set(secondary.applicability.get('keywords', []))
        merged_keywords = list(primary_keywords | secondary_keywords)
        primary.applicability['keywords'] = merged_keywords
        
        # Merge pitfalls
        primary_pitfall_patterns = {p['mistake'] for p in primary.common_pitfalls}
        for pitfall in secondary.common_pitfalls:
            if pitfall['mistake'] not in primary_pitfall_patterns:
                primary.common_pitfalls.append(pitfall)
        
        # Update version
        primary.version += 1
        primary.last_updated = datetime.now().isoformat()
        
        # Save merged strategy
        self.memory_store.store_strategy(primary)
        
        # Delete secondary
        self.memory_store.delete_strategy(secondary.strategy_id)
        
        # Log
        self.consolidation_history.append({
            'primary_id': primary.strategy_id,
            'secondary_id': secondary.strategy_id,
            'action': 'merge',
            'new_success_rate': primary.success_rate,
            'new_sample_count': primary.sample_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def _discover_new_patterns(
        self,
        trajectories: List[Trajectory],
        judgments: List[JudgmentResult],
        existing_strategies: List[ReasoningStrategy]
    ) -> List[ReasoningStrategy]:
        """Discover new patterns not covered by existing strategies"""
        
        # Get existing patterns
        existing_patterns = {s.pattern for s in existing_strategies}
        
        # Use distillation to find new strategies
        new_strategies = self.distillation.distill_strategies(
            trajectories,
            judgments
        )
        
        # Filter for truly new patterns
        novel_strategies = [
            s for s in new_strategies
            if s.pattern not in existing_patterns
        ]
        
        # Store novel strategies
        for strategy in novel_strategies:
            self.memory_store.store_strategy(strategy)
            logger.info(f"Discovered new pattern: {strategy.name}")
        
        return novel_strategies
    
    def optimize_memory(self) -> Dict[str, int]:
        """
        Optimize memory by consolidating and cleaning
        
        Returns:
            Statistics about optimization
        """
        logger.info("Optimizing memory...")
        
        stats = {
            'strategies_before': 0,
            'strategies_after': 0,
            'merged': 0,
            'deprecated': 0,
            'deleted': 0
        }
        
        # Get all strategies
        all_strategies = self.memory_store.get_all_strategies(active_only=False)
        stats['strategies_before'] = len(all_strategies)
        
        # Find merge opportunities
        stats['merged'] = self._find_and_merge_similar_strategies()
        
        # Remove low-performing strategies with insufficient data
        for strategy in all_strategies:
            if not self._is_strategy_viable(strategy):
                if strategy.sample_count < self.config['min_sample_count']:
                    self.memory_store.delete_strategy(strategy.strategy_id)
                    stats['deleted'] += 1
                else:
                    self.memory_store.deactivate_strategy(strategy.strategy_id)
                    stats['deprecated'] += 1
        
        # Count remaining
        remaining = self.memory_store.get_all_strategies(active_only=True)
        stats['strategies_after'] = len(remaining)
        
        logger.info(f"Memory optimization complete: {stats}")
        return stats
    
    def _is_strategy_viable(self, strategy: ReasoningStrategy) -> bool:
        """Check if strategy is viable to keep"""
        
        # Keep if high performance
        if strategy.success_rate >= 0.7:
            return True
        
        # Keep if reasonable performance and sufficient data
        if strategy.success_rate >= 0.5 and strategy.sample_count >= 20:
            return True
        
        # Keep if recently created (give it a chance)
        try:
            created = datetime.fromisoformat(strategy.created_at)
            age_days = (datetime.now() - created).days
            if age_days < 7:  # Less than a week old
                return True
        except:
            pass
        
        # Otherwise, not viable
        return False
    
    def get_consolidation_summary(self) -> Dict:
        """Get summary of consolidation activities"""
        
        if not self.consolidation_history:
            return {'message': 'No consolidation history'}
        
        # Count actions
        action_counts = Counter(h['action'] for h in self.consolidation_history)
        
        # Recent activities
        recent = self.consolidation_history[-10:]
        
        return {
            'total_consolidations': len(self.consolidation_history),
            'action_counts': dict(action_counts),
            'recent_activities': recent,
            'config': self.config
        }
    
    def export_consolidation_history(self, output_path: str):
        """Export consolidation history to file"""
        
        import json
        
        with open(output_path, 'w') as f:
            json.dump({
                'history': self.consolidation_history,
                'config': self.config,
                'exported_at': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Exported consolidation history to {output_path}")