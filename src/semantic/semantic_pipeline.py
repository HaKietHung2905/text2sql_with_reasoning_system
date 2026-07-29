"""
Semantic Pipeline Module
Proper pipeline integration for semantic layer
"""

from typing import Dict, List, Optional
import sys
from pathlib import Path

# Import semantic layer core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


try:
    from semantic_layer.core import SimpleSemanticLayer, create_semantic_layer
    SEMANTIC_CORE_AVAILABLE = True
except ImportError:
    SEMANTIC_CORE_AVAILABLE = False
    print("Warning: semantic_layer.core not available")

try:
    from utils.logging_utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SemanticPipeline:
    """
    Semantic Layer as Pipeline Component
    
    Converts semantic layer from wrapper pattern to proper pipeline step.
    Called BEFORE SQL generation to enhance questions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize semantic pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {'enabled': True}
        self.enabled = self.config.get('enabled', True)
        
        # Initialize semantic layer
        if SEMANTIC_CORE_AVAILABLE and self.enabled:
            try:
                self.semantic_layer = create_semantic_layer()
                logger.info("✓ Semantic Pipeline initialized")
            except Exception as e:
                logger.warning(f"Failed to create semantic layer: {e}")
                self.semantic_layer = None
                self.enabled = False
        else:
            self.semantic_layer = None
            self.enabled = False
            
        # Statistics
        self.stats = {
            'queries_analyzed': 0,
            'intents_detected': 0,
            'entities_mapped': 0,
            'enhanced_queries': 0
        }
    
    def enhance_question(
        self,
        question: str,
        db_id: Optional[str] = None,
        schema: Optional[Dict] = None,
    ) -> Dict:
        """
        Analyze a question with the rule-based semantic layer and produce
        a short natural-language hint block for prompt injection.

        The question text itself is NOT rewritten (safer — avoids corrupting
        the original NL question); instead, 'semantic_hints' carries the
        signal to be injected as a separate prompt block, analogous to how
        Semantic RAG contributes Ek(q) and ReasoningBank contributes Φm(q).

        Args:
            question: Natural language query
            db_id: Database identifier (kept for interface consistency;
                entity detection uses `schema` directly, not db_id)
            schema: {table_name: [column_names]} for the question's target
                database. When provided, entity detection generalizes to
                any database via SimpleSemanticLayer.detect_entities_from_schema.

        Returns:
            Dictionary with enhanced question and analysis
        """
        if not self.enabled or not self.semantic_layer:
            return {
                'original_question': question,
                'enhanced_question': question,
                'semantic_hints': '',
                'analysis': {},
                'enhanced': False
            }

        try:
            # Analyze query intent (schema-aware entity detection)
            analysis = self.semantic_layer.analyze_query_intent(
                question, schema_info=schema)

            # Track statistics
            self.stats['queries_analyzed'] += 1
            if analysis.get('relevant_metrics'):
                self.stats['intents_detected'] += 1
            if analysis.get('relevant_dimensions'):
                self.stats['entities_mapped'] += 1

            semantic_hints = self._format_semantic_hints(analysis)
            enhanced = bool(semantic_hints)
            if enhanced:
                self.stats['enhanced_queries'] += 1

            return {
                'original_question': question,
                'enhanced_question': question,
                'semantic_hints': semantic_hints,
                'analysis': analysis,
                'enhanced': enhanced,
                'complexity': self._assess_complexity(analysis)
            }

        except Exception as e:
            logger.warning(f"Question enhancement failed: {e}")
            return {
                'original_question': question,
                'enhanced_question': question,
                'semantic_hints': '',
                'analysis': {},
                'enhanced': False
            }

    def _format_semantic_hints(self, analysis: Dict) -> str:
        """Turn rule-based semantic analysis into short natural-language hints."""
        if not analysis:
            return ""

        hints = []
        intent_categories = analysis.get('intent_categories', [])
        metric_types = {m['type'] for m in analysis.get('relevant_metrics', [])}

        if 'count' in metric_types:
            hints.append("Likely needs COUNT aggregation.")
        if 'distinct_count' in metric_types:
            hints.append("Likely needs COUNT(DISTINCT ...).")
        if 'average' in metric_types:
            hints.append("Likely needs AVG aggregation.")
        if 'sum' in metric_types:
            hints.append("Likely needs SUM aggregation.")
        if 'max' in metric_types:
            hints.append("Likely needs MAX aggregation.")
        if 'min' in metric_types:
            hints.append("Likely needs MIN aggregation.")

        if analysis.get('relevant_dimensions') and 'grouping' in intent_categories:
            hints.append("Likely needs GROUP BY on a categorical or temporal column.")

        if 'ordering' in intent_categories:
            hints.append("Likely needs ORDER BY with LIMIT if a single top/bottom row is requested.")

        if 'filtering' in intent_categories or 'comparison' in intent_categories:
            hints.append("Likely needs a WHERE clause filtering on a specific value or comparison.")

        entities = analysis.get('relevant_entities', [])
        if entities:
            table_names = sorted({e.get('name', e.get('primary_table', '')) for e in entities if e})
            table_names = [t for t in table_names if t]
            if table_names:
                hints.append(f"Question likely references table(s): {', '.join(table_names)}.")

        if not hints:
            return ""

        return (
            "SEMANTIC LAYER HINTS (heuristic — verify against actual schema):\n"
            + "\n".join(f"- {h}" for h in hints) + "\n"
        )

    def analyze(self, question: str, schema: Optional[Dict] = None) -> Dict:
        """
        Alias for enhance_question - analyze a question
        
        This is the method expected by the integration test and other components.
        
        Args:
            question: Natural language query
            schema: Database schema (optional)
            
        Returns:
            Dictionary with analysis and complexity assessment
        """
        result = self.enhance_question(question, None, schema)
        
        # Return simplified format for analysis
        return {
            'original_question': result['original_question'],
            'enhanced_question': result['enhanced_question'],
            'complexity': result.get('complexity', 'medium'),
            'metrics': result.get('analysis', {}).get('relevant_metrics', []),
            'dimensions': result.get('analysis', {}).get('relevant_dimensions', []),
            'enhanced': result.get('enhanced', False)
        }
    
    def _assess_complexity(self, analysis: Dict) -> str:
        """
        Assess query complexity based on analysis
        
        Args:
            analysis: Query intent analysis
            
        Returns:
            Complexity level: 'easy', 'medium', or 'hard'
        """
        if not analysis:
            return 'medium'
        
        # Count complexity indicators
        num_metrics = len(analysis.get('relevant_metrics', []))
        num_dimensions = len(analysis.get('relevant_dimensions', []))
        has_aggregation = any(
            m.get('type') == 'AGGREGATION' 
            for m in analysis.get('relevant_metrics', [])
        )
        
        # Determine complexity
        if num_metrics == 0 and num_dimensions <= 1:
            return 'easy'
        elif num_metrics > 2 or num_dimensions > 3 or has_aggregation:
            return 'hard'
        else:
            return 'medium'
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'queries_analyzed': 0,
            'intents_detected': 0,
            'entities_mapped': 0,
            'enhanced_queries': 0
        }