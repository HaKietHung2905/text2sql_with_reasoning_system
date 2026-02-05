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

def __init__(self):
    self.analysis_cache = {} 

def analyze(self, question: str) -> Dict:
    cache_key = question.lower().strip()
    if cache_key in self.analysis_cache:
        return self.analysis_cache[cache_key]
    
    result = self._perform_analysis(question)
    self.analysis_cache[cache_key] = result
    return result

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
    
    def enhance_question(self, question: str, schema: Optional[Dict] = None) -> Dict:
        """
        Original method name - enhance a question with semantic understanding
        
        Args:
            question: Natural language query
            schema: Database schema (optional)
            
        Returns:
            Dictionary with enhanced question and analysis
        """
        if not self.enabled or not self.semantic_layer:
            return {
                'original_question': question,
                'enhanced_question': question,
                'analysis': {},
                'enhanced': False
            }
        
        try:
            # Analyze query intent
            analysis = self.semantic_layer.analyze_query_intent(question)
            
            # Track statistics
            self.stats['queries_analyzed'] += 1
            if analysis.get('relevant_metrics'):
                self.stats['intents_detected'] += 1
            if analysis.get('relevant_dimensions'):
                self.stats['entities_mapped'] += 1
            
            # Enhance question (placeholder - implement your enhancement logic)
            enhanced_question = question
            enhanced = False
            
            return {
                'original_question': question,
                'enhanced_question': enhanced_question,
                'analysis': analysis,
                'enhanced': enhanced,
                'complexity': self._assess_complexity(analysis)
            }
            
        except Exception as e:
            logger.warning(f"Question enhancement failed: {e}")
            return {
                'original_question': question,
                'enhanced_question': question,
                'analysis': {},
                'enhanced': False
            }
    
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
        result = self.enhance_question(question, schema)
        
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