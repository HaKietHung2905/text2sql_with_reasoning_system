"""
Memory Retrieval: Advanced strategy retrieval with context awareness

This module provides sophisticated retrieval mechanisms for reasoning strategies,
including semantic search, intent matching, and context-aware ranking.

Based on ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter

from .memory_store import ReasoningMemoryStore
from .strategy_distillation import ReasoningStrategy
from utils.logging_utils import get_logger
from utils.embedding_utils import EmbeddingGenerator

logger = get_logger(__name__)


@dataclass
class RetrievalContext:
    """Context information for strategy retrieval"""
    query: str                          # Natural language query
    database: str                       # Database name
    schema: Dict                        # Database schema
    semantic_analysis: Optional[Dict] = None  # Semantic layer output
    difficulty: Optional[str] = None    # Query difficulty estimate
    previous_strategies: Optional[List[str]] = None  # Recently used strategies


@dataclass
class RetrievalResult:
    """Result of strategy retrieval with metadata"""
    strategy: ReasoningStrategy
    relevance_score: float              # Overall relevance score
    similarity_score: float             # Semantic similarity
    intent_match_score: float           # Intent matching score
    performance_score: float            # Historical performance score
    recency_score: float                # How recently used/updated
    confidence: float                   # Confidence in retrieval
    reasoning: str                      # Why this strategy was retrieved


class MemoryRetrieval:
    """
    Advanced memory retrieval system
    
    Features:
    - Multi-faceted relevance scoring
    - Intent-based filtering
    - Context-aware ranking
    - Performance-weighted retrieval
    - Diversity-aware selection
    """
    
    def __init__(
        self,
        memory_store: ReasoningMemoryStore,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize memory retrieval
        
        Args:
            memory_store: ReasoningMemoryStore instance
            embedding_model: Sentence transformer model name
        """
        self.memory_store = memory_store
        self.embedding_gen = EmbeddingGenerator(embedding_model)
        
        # Retrieval configuration
        self.config = {
            'similarity_weight': 0.35,
            'intent_weight': 0.25,
            'performance_weight': 0.25,
            'recency_weight': 0.15,
            'min_similarity': 0.3,
            'min_success_rate': 0.0,
            'diversity_threshold': 0.85
        }
        
        logger.info("MemoryRetrieval initialized")
    
    def retrieve_strategies(
        self,
        context: RetrievalContext,
        top_k: int = 3,
        min_confidence: float = 0.5,
        ensure_diversity: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve most relevant strategies for given context
        
        Args:
            context: Retrieval context with query and metadata
            top_k: Number of strategies to retrieve
            min_confidence: Minimum confidence threshold
            ensure_diversity: Whether to ensure diverse strategies
            
        Returns:
            List of RetrievalResult objects
        """
        logger.debug(f"Retrieving strategies for: {context.query[:50]}...")
        
        # Step 1: Intent-based filtering
        candidate_patterns = self._identify_candidate_patterns(context)
        
        # Step 2: Get candidates from multiple sources
        candidates = self._get_candidates(context, candidate_patterns, top_k * 3)
        
        if not candidates:
            logger.warning("No candidate strategies found")
            return []
        
        # Step 3: Calculate multi-faceted relevance scores
        scored_candidates = []
        for strategy in candidates:
            result = self._calculate_relevance(strategy, context)
            if result.confidence >= min_confidence:
                scored_candidates.append(result)
        
        # Step 4: Rank and select top-k
        scored_candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Step 5: Ensure diversity if requested
        if ensure_diversity:
            selected = self._select_diverse_strategies(scored_candidates, top_k)
        else:
            selected = scored_candidates[:top_k]
        
        #logger.info(f"Retrieved {len(selected)} strategies (from {len(candidates)} candidates)")
        return selected
    
    def _identify_candidate_patterns(self, context: RetrievalContext) -> List[str]:
        """Identify potential SQL patterns from context"""
            
        patterns = set()
        query_lower = context.query.lower()
            
        # Detect aggregation patterns
        if any(word in query_lower for word in ['average', 'mean', 'avg', 'sum', 'total', 'count', 'how many', 'number of', 'max', 'maximum', 'min', 'minimum']):
            patterns.add('aggregation')
            
        # Detect grouping patterns
        if any(word in query_lower for word in ['each', 'every', 'per', 'by', 'group']):
            patterns.add('grouping')
        
        # Detect ranking patterns
        if any(word in query_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'first', 'last']):
            patterns.add('ranking')
            patterns.add('ordering')
        
        # Detect filtering patterns
        if any(word in query_lower for word in ['where', 'with', 'that', 'which', 'greater', 'less', 'equal', 'between']):
            patterns.add('filtering')
        
        # Detect join patterns from schema
        if context.schema and len(context.schema.get('tables', [])) > 1:
            # Check if query mentions multiple tables/entities
            table_names = [t.lower() for t in context.schema.get('tables', [])]
            mentioned_tables = sum(1 for table in table_names if table in query_lower)
            if mentioned_tables > 1:
                patterns.add('join')  # ← CHANGE THIS FROM 'JOIN' to 'join'
        
        # Use semantic analysis if available
        if context.semantic_analysis:
            intent = context.semantic_analysis.get('intent', '')
            if intent:
                # Extract patterns from intent (convert to lowercase)
                intent_patterns = [p.lower() for p in intent.split('_')]  # ← ADD .lower()
                patterns.update(intent_patterns)
        
        # Always include basic select as fallback
        if not patterns:
            patterns.add('basic_select')  # ← CHANGE THIS FROM 'BASIC_SELECT' to 'basic_select'
        
        return list(patterns)
    
    def _get_candidates(
        self,
        context: RetrievalContext,
        patterns: List[str],
        max_candidates: int
    ) -> List[ReasoningStrategy]:
        """Get candidate strategies from multiple sources"""
        
        candidates = []
        seen_ids = set()
        
        logger.debug(f"Looking for patterns: {patterns}")  # ← ADD THIS
        
        # Source 1: Pattern-based retrieval
        for pattern in patterns:
            strategies = self.memory_store.get_strategies_by_pattern(pattern)
            logger.debug(f"Pattern '{pattern}' found {len(strategies)} strategies")  # ← ADD THIS
            for strategy in strategies:
                if strategy.strategy_id not in seen_ids:
                    candidates.append(strategy)
                    seen_ids.add(strategy.strategy_id)
        
        logger.debug(f"After pattern matching: {len(candidates)} candidates")  # ← ADD THIS
        
        # Source 2: Semantic similarity search
        if len(candidates) < max_candidates:
            search_query = self._build_search_query(context)
            logger.debug(f"Semantic search query: {search_query[:100]}")  
            similar_strategies = self.memory_store.search_strategies(
                query=search_query,
                n_results=max_candidates,
                min_success_rate=0.0
            )
            logger.debug(f"Semantic search found {len(similar_strategies)} strategies")  
            
            for strategy, _ in similar_strategies:
                if strategy.strategy_id not in seen_ids:
                    candidates.append(strategy)
                    seen_ids.add(strategy.strategy_id)
        
        logger.debug(f"After semantic search: {len(candidates)} candidates")  
        
        # Source 3: High-performing strategies (if still need more)
        if len(candidates) < max_candidates:
            all_strategies = self.memory_store.get_all_strategies(
                min_success_rate=0.0
            )
            logger.debug(f"All strategies in DB: {len(all_strategies)}")
            
            for strategy in all_strategies:
                if strategy.strategy_id not in seen_ids:
                    candidates.append(strategy)
                    seen_ids.add(strategy.strategy_id)
                    if len(candidates) >= max_candidates:
                        break
        
        logger.debug(f"Final candidates: {len(candidates)}")  
        
        return candidates[:max_candidates]
    
    def _build_search_query(self, context: RetrievalContext) -> str:
        "Build search query from context"
    
        parts = [context.query]
        
        # Add semantic analysis if available
        if context.semantic_analysis:
            # Handle metrics (can be list of dicts or list of strings)
            metrics = context.semantic_analysis.get('relevant_metrics', [])
            if metrics:
                # Convert dicts to strings if needed
                metric_strs = []
                for m in metrics:
                    if isinstance(m, dict):
                        metric_strs.append(m.get('name', str(m)))
                    else:
                        metric_strs.append(str(m))
                
                if metric_strs:
                    parts.append(f"Metrics: {', '.join(metric_strs)}")
            
            # Handle dimensions (can be list of dicts or list of strings)
            dimensions = context.semantic_analysis.get('relevant_dimensions', [])
            if dimensions:
                # Convert dicts to strings if needed
                dim_strs = []
                for d in dimensions:
                    if isinstance(d, dict):
                        dim_strs.append(d.get('name', str(d)))
                    else:
                        dim_strs.append(str(d))
                
                if dim_strs:
                    parts.append(f"Dimensions: {', '.join(dim_strs)}")
        
        # Add database context
        if context.database:
            parts.append(f"Database: {context.database}")
        
        # Add difficulty if available
        if context.difficulty:
            parts.append(f"Difficulty: {context.difficulty}")
        
        return " | ".join(parts)
    
    def _calculate_relevance(
        self,
        strategy: ReasoningStrategy,
        context: RetrievalContext
    ) -> RetrievalResult:
        """Calculate multi-faceted relevance score"""
        
        # Component 1: Semantic similarity
        similarity_score = self._calculate_similarity(strategy, context)
        
        # Component 2: Intent matching
        intent_score = self._calculate_intent_match(strategy, context)
        
        # Component 3: Performance score
        performance_score = strategy.success_rate
        
        # Component 4: Recency score
        recency_score = self._calculate_recency_score(strategy)
        
        # Weighted combination
        relevance_score = (
            self.config['similarity_weight'] * similarity_score +
            self.config['intent_weight'] * intent_score +
            self.config['performance_weight'] * performance_score +
            self.config['recency_weight'] * recency_score
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            similarity_score,
            intent_score,
            performance_score,
            strategy.sample_count
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            strategy,
            similarity_score,
            intent_score,
            performance_score
        )
        
        return RetrievalResult(
            strategy=strategy,
            relevance_score=relevance_score,
            similarity_score=similarity_score,
            intent_match_score=intent_score,
            performance_score=performance_score,
            recency_score=recency_score,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _calculate_similarity(
        self,
        strategy: ReasoningStrategy,
        context: RetrievalContext
    ) -> float:
        """Calculate semantic similarity between strategy and query"""
        
        try:
            # Build query text
            query_text = self._build_search_query(context)
            
            # Build strategy text
            strategy_text = self._strategy_to_search_text(strategy)
            
            # Generate embeddings using the correct method
            if hasattr(self.embedding_gen, 'encode'):
                query_emb = self.embedding_gen.encode(query_text).tolist()
                strategy_emb = self.embedding_gen.encode(strategy_text).tolist()
            elif hasattr(self.embedding_gen, 'get_embedding'):
                query_emb = self.embedding_gen.get_embedding(query_text)
                strategy_emb = self.embedding_gen.get_embedding(strategy_text)
            else:
                # Fallback - return neutral similarity
                logger.warning("EmbeddingGenerator has no known embedding method")
                return 0.5
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_emb, strategy_emb)
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.5  # Return neutral score on error
    
    def _strategy_to_search_text(self, strategy: ReasoningStrategy) -> str:
        """Convert strategy to searchable text"""
        
        parts = [
            strategy.name,
            strategy.description,
            strategy.pattern,
            " ".join(strategy.reasoning_steps[:3]),  # First 3 steps
            " ".join(strategy.applicability.get('keywords', []))
        ]
        
        return " ".join(parts)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _calculate_intent_match(
        self,
        strategy: ReasoningStrategy,
        context: RetrievalContext
    ) -> float:
        """Calculate how well strategy matches query intent"""
        
        score = 0.0
        matches = 0
        total_checks = 0
        
        query_lower = context.query.lower()
        
        # Check 1: Keyword matching
        strategy_keywords = set(k.lower() for k in strategy.applicability.get('keywords', []))
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        keyword_overlap = len(strategy_keywords & query_words)
        if strategy_keywords:
            keyword_score = keyword_overlap / len(strategy_keywords)
            score += keyword_score
            total_checks += 1
        
        # Check 2: Intent type matching
        if context.semantic_analysis:
            context_intent = context.semantic_analysis.get('intent', '').lower()  # ← ADD .lower()
            strategy_intents = [i.lower() for i in strategy.applicability.get('intent_types', [])]  # ← ADD .lower()
            
            if context_intent and strategy_intents:
                # Check if any intent component matches
                context_parts = set(context_intent.split('_'))
                strategy_parts = set('_'.join(strategy_intents).split('_'))
                
                intent_overlap = len(context_parts & strategy_parts)
                if context_parts:
                    intent_score = intent_overlap / len(context_parts)
                    score += intent_score * 1.5  # Weight intent matching higher
                    total_checks += 1.5
        
        # Check 3: SQL pattern matching - NORMALIZE TO LOWERCASE
        strategy_patterns = set(p.lower() for p in strategy.applicability.get('sql_patterns', []))  # ← ADD .lower()
        
        # Detect required SQL patterns from query (use lowercase)
        required_patterns = set()
        if any(word in query_lower for word in ['join', 'combine', 'with', 'and']):
            required_patterns.add('join')  # ← CHANGE to lowercase
        if any(word in query_lower for word in ['group', 'each', 'per']):
            required_patterns.add('group by')  # ← CHANGE to lowercase
        if any(word in query_lower for word in ['order', 'sort', 'top', 'highest', 'lowest']):
            required_patterns.add('order by')  # ← CHANGE to lowercase
        if any(word in query_lower for word in ['limit', 'top', 'first', 'last']):
            required_patterns.add('limit')  # ← CHANGE to lowercase
        
        if required_patterns:
            pattern_overlap = len(required_patterns & strategy_patterns)
            pattern_score = pattern_overlap / len(required_patterns)
            score += pattern_score
            total_checks += 1
        
        # Check 4: Pattern exact match - NORMALIZE TO LOWERCASE
        if strategy.pattern:
            # Check if strategy pattern components match query needs
            pattern_parts = set(p.lower() for p in strategy.pattern.split('_'))  # ← ADD .lower()
            
            # Extract implied patterns from query (use lowercase)
            implied_patterns = set()
            if 'aggregat' in query_lower or 'average' in query_lower or 'sum' in query_lower:
                implied_patterns.add('aggregation')  # ← CHANGE to lowercase
            if 'each' in query_lower or 'per' in query_lower:
                implied_patterns.add('grouping')  # ← CHANGE to lowercase
            if 'top' in query_lower or 'rank' in query_lower:
                implied_patterns.add('ranking')  # ← CHANGE to lowercase
            
            if implied_patterns:
                exact_match = len(pattern_parts & implied_patterns)
                if pattern_parts:
                    exact_score = exact_match / len(pattern_parts)
                    score += exact_score
                    total_checks += 1
        
        # Normalize score
        if total_checks > 0:
            return score / total_checks
        
        return 0.0
    
    def _calculate_recency_score(self, strategy: ReasoningStrategy) -> float:
        """Calculate recency score based on last update time"""
        
        try:
            from datetime import datetime
            
            last_updated = datetime.fromisoformat(strategy.last_updated)
            now = datetime.now()
            
            # Days since last update
            days_old = (now - last_updated).days
            
            # Decay function: 1.0 at 0 days, 0.5 at 30 days, 0.1 at 90 days
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 0.8
            elif days_old <= 60:
                return 0.6
            elif days_old <= 90:
                return 0.4
            else:
                return 0.2
        
        except Exception as e:
            logger.debug(f"Could not calculate recency: {e}")
            return 0.5  # Neutral score
    
    def _calculate_confidence(
        self,
        similarity: float,
        intent_match: float,
        performance: float,
        sample_count: int
    ) -> float:
        """Calculate confidence in retrieval"""
        
        # Base confidence from scores
        base_confidence = (similarity + intent_match + performance) / 3
        
        # Adjust based on sample count
        if sample_count < 5:
            sample_factor = 0.7  # Low confidence with few samples
        elif sample_count < 20:
            sample_factor = 0.85
        elif sample_count < 50:
            sample_factor = 0.95
        else:
            sample_factor = 1.0  # High confidence with many samples
        
        confidence = base_confidence * sample_factor
        
        # Boost confidence if multiple signals are strong
        strong_signals = sum([
            similarity > 0.8,
            intent_match > 0.7,
            performance > 0.8
        ])
        
        if strong_signals >= 2:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence
    
    def _generate_reasoning(
        self,
        strategy: ReasoningStrategy,
        similarity: float,
        intent_match: float,
        performance: float
    ) -> str:
        """Generate human-readable reasoning for retrieval"""
        
        reasons = []
        
        # Similarity reasoning
        if similarity > 0.8:
            reasons.append(f"High semantic similarity ({similarity:.1%})")
        elif similarity > 0.6:
            reasons.append(f"Good semantic match ({similarity:.1%})")
        
        # Intent reasoning
        if intent_match > 0.7:
            reasons.append(f"Strong intent alignment ({intent_match:.1%})")
        elif intent_match > 0.5:
            reasons.append(f"Matching query intent ({intent_match:.1%})")
        
        # Performance reasoning
        if performance > 0.8:
            reasons.append(f"Proven success rate ({performance:.1%})")
        elif performance > 0.6:
            reasons.append(f"Good track record ({performance:.1%})")
        
        # Sample count reasoning
        if strategy.sample_count > 50:
            reasons.append(f"Extensively validated ({strategy.sample_count} examples)")
        elif strategy.sample_count > 20:
            reasons.append(f"Well-tested ({strategy.sample_count} examples)")
        
        if reasons:
            return " | ".join(reasons)
        else:
            return "Potential match based on pattern"
    
    def _select_diverse_strategies(
        self,
        candidates: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Select diverse strategies to avoid redundancy"""
        
        if len(candidates) <= top_k:
            return candidates
        
        selected = [candidates[0]]  # Always take the top candidate
        
        for candidate in candidates[1:]:
            if len(selected) >= top_k:
                break
            
            # Check diversity with already selected strategies
            is_diverse = True
            for selected_result in selected:
                if self._strategies_too_similar(
                    candidate.strategy,
                    selected_result.strategy
                ):
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        # If we don't have enough diverse strategies, fill with top candidates
        if len(selected) < top_k:
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    if len(selected) >= top_k:
                        break
        
        return selected
    
    def _strategies_too_similar(
        self,
        strategy1: ReasoningStrategy,
        strategy2: ReasoningStrategy
    ) -> bool:
        """Check if two strategies are too similar"""
        
        # Same pattern = too similar
        if strategy1.pattern == strategy2.pattern:
            return True
        
        # Check keyword overlap
        keywords1 = set(strategy1.applicability.get('keywords', []))
        keywords2 = set(strategy2.applicability.get('keywords', []))
        
        if keywords1 and keywords2:
            overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            if overlap > self.config['diversity_threshold']:
                return True
        
        return False
    
    def retrieve_by_pattern(
        self,
        pattern: str,
        top_k: int = 3
    ) -> List[ReasoningStrategy]:
        """Retrieve strategies by exact pattern match"""
        
        strategies = self.memory_store.get_strategies_by_pattern(pattern)
        
        # Sort by success rate
        strategies.sort(key=lambda s: s.success_rate, reverse=True)
        
        return strategies[:top_k]
    
    def retrieve_similar_to_strategy(
        self,
        strategy_id: str,
        top_k: int = 5
    ) -> List[Tuple[ReasoningStrategy, float]]:
        """Find strategies similar to a given strategy"""
        
        # Get the reference strategy
        ref_strategy = self.memory_store.get_strategy(strategy_id)
        if not ref_strategy:
            return []
        
        # Search using strategy description
        search_text = self._strategy_to_search_text(ref_strategy)
        
        results = self.memory_store.search_strategies(
            query=search_text,
            n_results=top_k + 1  # +1 because reference will be included
        )
        
        # Filter out the reference strategy itself
        filtered = [
            (s, score) for s, score in results
            if s.strategy_id != strategy_id
        ]
        
        return filtered[:top_k]
    
    def get_complementary_strategies(
        self,
        strategy_id: str,
        top_k: int = 3
    ) -> List[ReasoningStrategy]:
        """Get strategies that complement the given strategy"""
        
        # This would query strategy_relationships table
        # For now, return strategies with different patterns
        
        ref_strategy = self.memory_store.get_strategy(strategy_id)
        if not ref_strategy:
            return []
        
        # Get strategies with different patterns
        all_strategies = self.memory_store.get_all_strategies()
        
        # Filter for different patterns but potentially complementary
        complementary = []
        ref_pattern_parts = set(ref_strategy.pattern.split('_'))
        
        for strategy in all_strategies:
            if strategy.strategy_id == strategy_id:
                continue
            
            pattern_parts = set(strategy.pattern.split('_'))
            
            # Check if patterns have some overlap but are not identical
            overlap = len(ref_pattern_parts & pattern_parts)
            if 0 < overlap < len(ref_pattern_parts):
                complementary.append(strategy)
        
        # Sort by success rate
        complementary.sort(key=lambda s: s.success_rate, reverse=True)
        
        return complementary[:top_k]
    
    def format_for_prompt(self, results: List[RetrievalResult]) -> str:
        """Format retrieved strategies for LLM prompt"""
        
        if not results:
            return ""
        
        formatted_parts = []
        
        for i, result in enumerate(results, 1):
            strategy = result.strategy
            
            formatted = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY #{i}: {strategy.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Performance: {strategy.success_rate:.1%} success rate ({strategy.sample_count} examples)
🎯 Relevance: {result.relevance_score:.1%}
💡 Why selected: {result.reasoning}

WHEN TO APPLY:
{chr(10).join(f"  ✓ {kw}" for kw in strategy.applicability.get('keywords', [])[:5])}

REASONING STEPS:
{chr(10).join(f"  {step}" for step in strategy.reasoning_steps)}

CRITICAL RULES:
{chr(10).join(f"  ⚠️  {rule}" for rule in strategy.critical_rules)}

COMMON MISTAKES TO AVOID:
{chr(10).join(f"  ❌ {p['mistake']} → ✅ {p['fix']}" for p in strategy.common_pitfalls[:3])}

SQL STRUCTURE HINT:
{strategy.sql_template_hints.get('structure', 'N/A')}
"""
            formatted_parts.append(formatted)
        
        return "\n".join(formatted_parts)
    
    def get_retrieval_statistics(self) -> Dict:
        """Get statistics about retrieval performance"""
        
        # This would track retrieval history
        # For now, return basic stats
        
        return {
            'total_strategies': len(self.memory_store.get_all_strategies()),
            'avg_success_rate': self.memory_store.get_statistics().get('avg_success_rate', 0.0),
            'config': self.config
        }