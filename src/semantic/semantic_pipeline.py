"""
Semantic Pipeline Module
Proper pipeline integration for semantic layer
"""

from typing import Dict, List, Optional
import sys
from pathlib import Path

# Import semantic layer core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from semantic_layer.core import SimpleSemanticLayer, create_semantic_layer
    SEMANTIC_CORE_AVAILABLE = True
except ImportError:
    SEMANTIC_CORE_AVAILABLE = False
    print("Warning: semantic_layer.core not available")

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SemanticPipeline:
    """
    Semantic Layer as Pipeline Component
    
    Converts semantic layer from wrapper pattern to proper pipeline step.
    Called BEFORE SQL generation to enhance questions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize semantic pipeline"""
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled or not SEMANTIC_CORE_AVAILABLE:
            logger.info("Semantic Pipeline disabled")
            self.semantic_layer = None
            return
        
        # Initialize semantic layer
        config_path = self.config.get('config_path')
        if config_path and Path(config_path).exists():
            self.semantic_layer = SimpleSemanticLayer(config_path)
        else:
            self.semantic_layer = create_semantic_layer()
        
        # Statistics
        self.stats = {
            'queries_analyzed': 0,
            'queries_enhanced': 0,
            'intents_detected': {},
            'metrics_detected': 0,
            'dimensions_detected': 0,
            'entities_detected': 0
        }
        
        logger.info("✓ Semantic Pipeline initialized")
    
    def enhance_question(
        self,
        question: str,
        database: str,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Main enhancement method - called BEFORE SQL generation
        
        Returns dict with:
            - enhanced_question: Enhanced question text
            - suggestions: SQL generation suggestions
            - intent: Detected intent
            - metrics/dimensions/entities: Detected components
            - enhanced: Whether enhancement was applied
        """
        if not self.enabled or not self.semantic_layer:
            return self._no_enhancement(question)
        
        self.stats['queries_analyzed'] += 1
        
        try:
            # Analyze intent
            analysis = self.semantic_layer.analyze_query_intent(question)
            context = self.semantic_layer.get_semantic_context(question, schema)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(analysis, context)
            
            # Update stats
            self._update_stats(analysis)
            
            # Check if enhanced
            enhanced = (
                len(analysis['relevant_metrics']) > 0 or
                len(analysis['relevant_dimensions']) > 0
            )
            
            if enhanced:
                self.stats['queries_enhanced'] += 1
            
            return {
                'enhanced_question': question,  # Could add semantic hints here
                'original_question': question,
                'suggestions': suggestions,
                'intent': context.get('intent'),
                'metrics': analysis['relevant_metrics'],
                'dimensions': analysis['relevant_dimensions'],
                'entities': analysis['relevant_entities'],
                'complexity': context.get('complexity'),
                'enhanced': enhanced
            }
            
        except Exception as e:
            logger.error(f"Semantic enhancement failed: {e}")
            return self._no_enhancement(question)
    
    def _no_enhancement(self, question: str) -> Dict:
        """Return when enhancement disabled/fails"""
        return {
            'enhanced_question': question,
            'original_question': question,
            'suggestions': [],
            'intent': None,
            'metrics': [],
            'dimensions': [],
            'entities': [],
            'complexity': 'unknown',
            'enhanced': False
        }
    
    def _generate_suggestions(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Generate SQL suggestions"""
        suggestions = []
        
        # From SQL patterns
        for pattern in analysis.get('suggested_sql_patterns', [])[:3]:
            suggestions.append({
                'type': 'pattern',
                'content': pattern,
                'priority': 'medium'
            })
        
        # From intent
        for intent in analysis.get('intent_categories', []):
            if intent == 'aggregation':
                suggestions.append({
                    'type': 'aggregation',
                    'content': 'Use COUNT/SUM/AVG/MAX/MIN',
                    'priority': 'high'
                })
            elif intent == 'grouping':
                suggestions.append({
                    'type': 'grouping',
                    'content': 'Add GROUP BY clause',
                    'priority': 'high'
                })
        
        return suggestions
    
    def _update_stats(self, analysis: Dict):
        """Update statistics"""
        for intent in analysis.get('intent_categories', []):
            self.stats['intents_detected'][intent] = \
                self.stats['intents_detected'].get(intent, 0) + 1
        
        self.stats['metrics_detected'] += len(analysis.get('relevant_metrics', []))
        self.stats['dimensions_detected'] += len(analysis.get('relevant_dimensions', []))
        self.stats['entities_detected'] += len(analysis.get('relevant_entities', []))
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        stats = {'enabled': True, **self.stats}
        
        if stats['queries_analyzed'] > 0:
            stats['enhancement_rate'] = \
                stats['queries_enhanced'] / stats['queries_analyzed'] * 100
        else:
            stats['enhancement_rate'] = 0
        
        return stats
