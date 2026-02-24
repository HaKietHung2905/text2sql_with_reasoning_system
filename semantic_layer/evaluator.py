"""
Semantic Enhanced Evaluator Module
==================================

Enhanced evaluator that extends the existing evaluation system with semantic understanding
capabilities for improved Text-to-SQL generation.
"""

import os
import sys
import sqlite3
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Add parent directory to path to enable imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the semantic layer core
try:
    from .core import SimpleSemanticLayer, create_semantic_layer, enhance_sql_generation
    SEMANTIC_CORE_AVAILABLE = True
except ImportError:
    try:
        from core import SimpleSemanticLayer, create_semantic_layer, enhance_sql_generation
        SEMANTIC_CORE_AVAILABLE = True
    except ImportError:
        print("Warning: Semantic layer core not available")
        SEMANTIC_CORE_AVAILABLE = False
        SimpleSemanticLayer = None

# Try to import existing evaluator - FIXED IMPORT
try:
    # Import from utils.eval with proper base classes
    from utils.eval import BaseEvaluator, ChromaDBEvaluator
    # Use ChromaDBEvaluator as parent for best functionality
    Evaluator = ChromaDBEvaluator
    EXISTING_EVAL_AVAILABLE = True
    print("✓ Successfully imported ChromaDBEvaluator from utils.eval")
except ImportError:
    try:
        # Fallback to BaseEvaluator
        from utils.eval import BaseEvaluator
        Evaluator = BaseEvaluator
        EXISTING_EVAL_AVAILABLE = True
        print("✓ Successfully imported BaseEvaluator from utils.eval")
    except ImportError:
        print("⚠️  Could not import from utils.eval. Semantic layer will have limited functionality.")
        EXISTING_EVAL_AVAILABLE = False
        # Create a minimal base class with essential methods
        class Evaluator:
            def __init__(self, *args, **kwargs):
                self.partial_scores = None
                self.langchain_generator = None
            
            def _get_db_schema(self, db_path):
                """Minimal schema getter"""
                schema_info = {}
                if not os.path.exists(db_path):
                    return schema_info
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [table[0] for table in cursor.fetchall()]
                    for table in tables:
                        cursor.execute(f'PRAGMA table_info("{table}")')
                        columns = [col[1] for col in cursor.fetchall()]
                        schema_info[table] = columns
                    conn.close()
                except Exception as e:
                    print(f"Error getting schema: {e}")
                return schema_info

class SemanticEvaluator():
    """
    Enhanced evaluator that adds semantic understanding to SQL generation.
    Extends the existing Evaluator class with semantic layer capabilities.
    """
    
    def __init__(self, prompt_type="enhanced", enable_debugging=False, 
             use_chromadb=False, chromadb_config=None, semantic_config_path=None):
    
        # Dynamically import at runtime with proper path handling
        import sys
        from pathlib import Path
        
        # Ensure project root is in path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        sys.path.insert(0, str(project_root))
        
        # Now import - this WILL work
        self._base_evaluator = None
        
        # Import the actual classes from utils.eval module
        import importlib.util
        eval_module_path = project_root / 'utils' / 'eval.py'
        
        if not eval_module_path.exists():
            raise RuntimeError(f"Cannot find utils/eval.py at {eval_module_path}")
        
        # Load the module directly
        spec = importlib.util.spec_from_file_location("utils.eval", eval_module_path)
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        
        # Create the evaluator
        if hasattr(eval_module, 'ChromaDBEvaluator'):
            self._base_evaluator = eval_module.ChromaDBEvaluator(
                prompt_type, enable_debugging, use_chromadb, chromadb_config
            )
            print("✓ Using ChromaDBEvaluator")
        elif hasattr(eval_module, 'BaseEvaluator'):
            self._base_evaluator = eval_module.BaseEvaluator()
            print("✓ Using BaseEvaluator")
        else:
            raise RuntimeError("No evaluator classes found in utils/eval.py")
        
        # Initialize semantic layer
        if SEMANTIC_CORE_AVAILABLE:
            try:
                if semantic_config_path and os.path.exists(semantic_config_path):
                    self.semantic_layer = SimpleSemanticLayer(semantic_config_path)
                else:
                    self.semantic_layer = create_semantic_layer()
                self.semantic_enabled = True
                print("✓ Semantic layer initialized")
            except Exception as e:
                print(f"✗ Semantic layer failed: {e}")
                self.semantic_layer = None
                self.semantic_enabled = False
        else:
            self.semantic_layer = None
            self.semantic_enabled = False
        
        # Statistics tracking
        self.semantic_stats = {
            'queries_analyzed': 0,
            'queries_enhanced': 0,
            'suggestions_made': 0,
            'complexity_scores': [],
            'enhancement_types': {
                'count_enhancements': 0,
                'aggregation_enhancements': 0,
                'grouping_enhancements': 0,
                'join_enhancements': 0,
                'filtering_enhancements': 0,
                'ordering_enhancements': 0
            },
            'entity_detections': {
                'car_entities': 0,
                'student_entities': 0,
                'customer_entities': 0,
                'product_entities': 0,
                'other_entities': 0
            }
        }
        
        # Enhancement configuration
        self.enhancement_config = {
            'enable_count_enhancement': False,
            'enable_aggregation_enhancement': False,
            'enable_grouping_enhancement': False,
            'enable_join_suggestions': False,
            'enable_filtering_enhancement': False,
            'enable_ordering_enhancement': False,
            'max_suggestions_per_query': 5
        }
        
    def generate_sql_from_question(self, question: str, db_path: str) -> str:
        """Generate SQL using base evaluator"""
        
        # Use composition instead of inheritance
        if self._base_evaluator and hasattr(self._base_evaluator, 'generate_sql_from_question'):
            base_sql = self._base_evaluator.generate_sql_from_question(question, db_path)
        else:
            # Try to use Gemini SQLGenerator
            try:
                from src.generation.sql_generator import SQLGenerator
                gen = SQLGenerator()
                base_sql = gen.generate(question, db_path)
            except Exception as e:
                print(f"Gemini generation failed, using fallback: {e}")
                base_sql = self._simple_sql_generation(question, db_path)
        
        # Analyze for statistics only
        if self.semantic_enabled:
            try:
                self._analyze_for_statistics(question, base_sql, db_path)
            except Exception as e:
                print(f"Semantic analysis failed: {e}")
        
        self.semantic_stats['queries_analyzed'] += 1
        return base_sql
    
    def _analyze_for_statistics(self, question: str, sql: str, db_path: str):
        """Analyze query for statistics only - does not modify SQL"""
        try:
            schema_info = self._get_database_schema(db_path)
            analysis = self.semantic_layer.analyze_query_intent(question)
            
            # Track complexity
            self.semantic_stats['complexity_scores'].append(analysis['complexity_score'])
            
            # Track entity detections
            self._track_entity_detections(analysis)
            
        except Exception as e:
            print(f"Statistics tracking error: {e}")
            
    def eval_exact_match(self, pred: Dict, gold: Dict) -> int:
        """Forward exact match evaluation to base evaluator"""
        if self._base_evaluator and hasattr(self._base_evaluator, 'eval_exact_match'):
            return self._base_evaluator.eval_exact_match(pred, gold)
        return 0
        
    @property
    def partial_scores(self) -> Dict:
        """Forward partial scores to base evaluator"""
        if self._base_evaluator and hasattr(self._base_evaluator, 'partial_scores'):
            return self._base_evaluator.partial_scores
        return None
    
    def _simple_sql_generation(self, question: str, db_path: str) -> str:
        """Simple SQL generation fallback"""
        schema_info = self._get_database_schema(db_path)
        question_lower = question.lower()
        
        # Get first table as default
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT 1"
        
        main_table = tables[0]
        
        # Simple pattern matching
        if any(word in question_lower for word in ['how many', 'count', 'number']):
            return f"SELECT COUNT(*) FROM {main_table}"
        elif any(word in question_lower for word in ['list', 'show', 'all']):
            return f"SELECT * FROM {main_table}"
        
        return f"SELECT * FROM {main_table}"
    
    def _get_database_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Get database schema information"""
        # Try parent method first
        if hasattr(super(), '_get_db_schema'):
            try:
                return super()._get_db_schema(db_path)
            except:
                pass
        
        # Fallback implementation
        schema_info = {}
        if not os.path.exists(db_path):
            return schema_info
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                schema_info[table] = columns
            
            conn.close()
        except Exception as e:
            print(f"Error getting schema: {e}")
        
        return schema_info
    
    def _track_entity_detections(self, analysis: Dict):
        """Track entity detections for statistics"""
        entities = analysis.get('relevant_entities', [])
        
        for entity in entities:
            entity_name = entity.get('name', '').lower()
            if 'car' in entity_name:
                self.semantic_stats['entity_detections']['car_entities'] += 1
            elif 'student' in entity_name:
                self.semantic_stats['entity_detections']['student_entities'] += 1
            elif 'customer' in entity_name:
                self.semantic_stats['entity_detections']['customer_entities'] += 1
            elif 'product' in entity_name:
                self.semantic_stats['entity_detections']['product_entities'] += 1
            else:
                self.semantic_stats['entity_detections']['other_entities'] += 1
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze a question for semantic understanding"""
        if not self.semantic_enabled:
            return {'error': 'Semantic layer not available'}
        
        try:
            analysis = self.semantic_layer.analyze_query_intent(question)
            return {
                'intent_analysis': analysis,
                'complexity': analysis.get('complexity_score', 0),
                'metrics_found': len(analysis.get('relevant_metrics', [])),
                'dimensions_found': len(analysis.get('relevant_dimensions', [])),
                'entities_found': len(analysis.get('relevant_entities', []))
            }
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive semantic enhancement statistics"""
        stats = self.semantic_stats.copy()
        
        # Calculate derived statistics
        if stats['queries_analyzed'] > 0:
            stats['enhancement_rate'] = stats['queries_enhanced'] / stats['queries_analyzed'] * 100
            stats['avg_suggestions'] = stats['suggestions_made'] / stats['queries_analyzed']
            
            if stats['complexity_scores']:
                stats['avg_complexity'] = sum(stats['complexity_scores']) / len(stats['complexity_scores'])
                stats['max_complexity'] = max(stats['complexity_scores'])
                stats['min_complexity'] = min(stats['complexity_scores'])
        else:
            stats['enhancement_rate'] = 0
            stats['avg_suggestions'] = 0
            stats['avg_complexity'] = 0
            stats['max_complexity'] = 0
            stats['min_complexity'] = 0
        
        # Add semantic layer info
        if self.semantic_enabled:
            stats['semantic_layer_status'] = 'active'
            stats['total_metrics'] = len(self.semantic_layer.metrics)
            stats['total_dimensions'] = len(self.semantic_layer.dimensions)
            stats['total_entities'] = len(self.semantic_layer.entities)
        else:
            stats['semantic_layer_status'] = 'inactive'
            stats['total_metrics'] = 0
            stats['total_dimensions'] = 0
            stats['total_entities'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.semantic_stats['queries_analyzed'] = 0
        self.semantic_stats['queries_enhanced'] = 0
        self.semantic_stats['suggestions_made'] = 0
        self.semantic_stats['complexity_scores'] = []
        for key in self.semantic_stats['enhancement_types']:
            self.semantic_stats['enhancement_types'][key] = 0
        for key in self.semantic_stats['entity_detections']:
            self.semantic_stats['entity_detections'][key] = 0


# Convenience functions
def evaluate_with_semantics(question: str, db_path: str, 
                          semantic_config_path: str = None) -> Dict[str, Any]:
    """Simple function to evaluate a question with semantic enhancement"""
    evaluator = SemanticEvaluator(semantic_config_path=semantic_config_path)
    sql = evaluator.generate_sql_from_question(question, db_path)
    analysis = evaluator.analyze_question(question)
    return {
        'question': question,
        'sql': sql,
        'analysis': analysis
    }

def batch_evaluate_with_semantics(questions: List[str], db_path: str,
                                 semantic_config_path: str = None) -> List[Dict[str, Any]]:
    """Evaluate multiple questions with semantic enhancement"""
    evaluator = SemanticEvaluator(semantic_config_path=semantic_config_path)
    results = []
    
    for question in questions:
        sql = evaluator.generate_sql_from_question(question, db_path)
        analysis = evaluator.analyze_question(question)
        results.append({
            'question': question,
            'sql': sql,
            'analysis': analysis
        })
    
    return results