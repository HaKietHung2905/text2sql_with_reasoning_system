"""
Semantic Layer Package for Text-to-SQL Enhancement
==================================================

This package provides semantic understanding capabilities to enhance
SQL generation from natural language queries.

Usage:
    from semantic_layer import SemanticEvaluator, create_semantic_layer
    
    # Create semantic layer
    semantic_layer = create_semantic_layer()
    
    # Analyze query intent
    analysis = semantic_layer.analyze_query_intent("How many cars?")
    
    # Use enhanced evaluator
    evaluator = SemanticEvaluator()
    result = evaluator.generate_enhanced_sql(question, db_path)
"""

# Import main classes and functions for easy access
try:
    from .core import (
        SimpleSemanticLayer,
        Metric,
        Dimension, 
        Entity,
        MetricType,
        DimensionType,
        create_semantic_layer,
        enhance_sql_generation
    )
    
    from .evaluator import (
        SemanticEvaluator,
        evaluate_with_semantics,
        batch_evaluate_with_semantics
    )
    
    # Package metadata
    __version__ = "1.0.0"
    __author__ = "Text-to-SQL Semantic Layer"
    __description__ = "Semantic layer for enhanced SQL generation from natural language"
    
    # Define what gets imported with "from semantic_layer import *"
    __all__ = [
        # Core classes
        "SimpleSemanticLayer",
        "Metric", 
        "Dimension",
        "Entity",
        "MetricType",
        "DimensionType",
        
        # Main functions
        "create_semantic_layer",
        "enhance_sql_generation",
        
        # Enhanced evaluator
        "SemanticEvaluator",
        "evaluate_with_semantics",
        "batch_evaluate_with_semantics",
    ]
    
    # Package status
    _SEMANTIC_LAYER_READY = True
    
except ImportError as e:
    # Graceful degradation if dependencies are missing
    print(f"Warning: Semantic layer import failed: {e}")
    _SEMANTIC_LAYER_READY = False
    
    # Provide dummy classes/functions for compatibility
    class SemanticEvaluator:
        def __init__(self, *args, **kwargs):
            raise ImportError("Semantic layer not available. Check installation.")
    
    def create_semantic_layer():
        raise ImportError("Semantic layer not available. Check installation.")
    
    def evaluate_with_semantics(*args, **kwargs):
        raise ImportError("Semantic layer not available. Check installation.")
    
    def batch_evaluate_with_semantics(*args, **kwargs):
        raise ImportError("Semantic layer not available. Check installation.")

def is_available():
    """Check if semantic layer is properly installed and available"""
    return _SEMANTIC_LAYER_READY

def get_version():
    """Get semantic layer version"""
    return __version__ if _SEMANTIC_LAYER_READY else "0.0.0"

def get_info():
    """Get package information"""
    return {
        "name": "semantic_layer",
        "version": __version__ if _SEMANTIC_LAYER_READY else "0.0.0",
        "description": __description__ if _SEMANTIC_LAYER_READY else "Not available",
        "author": __author__ if _SEMANTIC_LAYER_READY else "Unknown",
        "available": _SEMANTIC_LAYER_READY
    }

# Quick setup function for first-time users
def quick_setup():
    """
    Quick setup function to test semantic layer installation
    """
    print("🎯 Semantic Layer Quick Setup")
    print("=" * 30)
    
    if not is_available():
        print("❌ Semantic layer not available")
        print("Make sure all files are in the semantic_layer/ folder:")
        print("  - __init__.py (this file)")
        print("  - core.py")
        print("  - evaluator.py")
        print("  - config.json")
        return False
    
    print("✅ Semantic layer imported successfully")
    print(f"   Version: {get_version()}")
    
    # Test basic functionality
    try:
        semantic_layer = create_semantic_layer()
        test_question = "How many cars have horsepower greater than 200?"
        analysis = semantic_layer.analyze_query_intent(test_question)
        
        print(f"✅ Test analysis completed")
        print(f"   Question: {test_question}")
        print(f"   Metrics found: {len(analysis['relevant_metrics'])}")
        print(f"   Dimensions found: {len(analysis['relevant_dimensions'])}")
        print(f"   Complexity: {semantic_layer._assess_complexity(analysis)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# Quick test function
def test():
    """Run quick tests"""
    return quick_setup()

# Display info when package is imported
if _SEMANTIC_LAYER_READY:
    print(f"✅ Semantic Layer v{__version__} loaded successfully")