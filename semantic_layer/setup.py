#!/usr/bin/env python3
"""
Semantic Layer Setup and Validation Script
==========================================

This script validates and tests the semantic layer installation.

Usage:
    python semantic_layer/setup.py [--test] [--examples] [--all]
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to enable imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def check_package_structure():
    """Check if semantic layer package structure is correct"""
    
    print("Checking semantic layer package structure...")
    
    required_files = [
        '__init__.py',
        'core.py', 
        'evaluator.py',
        'config.json'
    ]
    
    current_dir = Path(__file__).parent
    missing_files = []
    
    for file_name in required_files:
        file_path = current_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"Missing files in semantic_layer/: {missing_files}")
        return False
    
    print("Package structure is correct")
    return True

def validate_config():
    """Validate the config.json file"""
    
    print("Validating config.json...")
    
    current_dir = Path(__file__).parent
    config_file = current_dir / "config.json"
    
    if not config_file.exists():
        print("config.json not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for required sections
        required_sections = ['metrics', 'dimensions', 'entities']
        config_data = config.get('semantic_layer_config', {})
        
        for section in required_sections:
            if section not in config_data:
                print(f"Missing required section: {section}")
                return False
        
        print(f"Config validated - {len(config_data.get('entities', {}))} entities configured")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Config JSON is invalid: {e}")
        return False
    except Exception as e:
        print(f"Config validation failed: {e}")
        return False

def validate_imports():
    """Validate that all imports work correctly"""
    
    print("Validating imports...")
    
    try:
        # Use absolute imports instead of relative imports
        import semantic_layer.core as core
        print("Core module imported")
        
        import semantic_layer.evaluator as evaluator
        print("Evaluator module imported")
        
        # Test main classes
        from semantic_layer.core import SimpleSemanticLayer, create_semantic_layer
        print("Core classes imported")
        
        from semantic_layer.evaluator import SemanticEvaluator
        print("Enhanced evaluator imported")
        
        # Test package-level imports
        from semantic_layer import SemanticEvaluator as PackageEvaluator
        print("Package-level imports work")
        
        return True
        
    except Exception as e:
        print(f"Import validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_functionality():
    """Test basic semantic layer functionality"""
    
    print("Testing semantic layer functionality...")
    
    try:
        from semantic_layer.core import create_semantic_layer
        
        # Create semantic layer
        semantic_layer = create_semantic_layer()
        print("Semantic layer created")
        
        # Test intent analysis
        test_questions = [
            "How many cars were made in 1980?",
            "What is the average horsepower?", 
            "Show me the highest MPG by manufacturer",
            "List students by department"
        ]
        
        for question in test_questions:
            analysis = semantic_layer.analyze_query_intent(question)
            
            metrics_count = len(analysis['relevant_metrics'])
            dimensions_count = len(analysis['relevant_dimensions'])
            complexity = semantic_layer._assess_complexity(analysis)
            
            print(f"  ✓ '{question[:40]}...' -> {metrics_count}M, {dimensions_count}D, {complexity}")
        
        print("Functionality tests passed")
        return True
        
    except Exception as e:
        print(f"Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator_integration():
    """Test enhanced evaluator functionality"""
    
    print("Testing enhanced evaluator...")
    
    try:
        from semantic_layer.evaluator import SemanticEvaluator
        
        # Create evaluator
        evaluator = SemanticEvaluator()
        print("Enhanced evaluator created")
        
        # Test analysis function
        test_question = "How many students are enrolled?"
        analysis = evaluator.analyze_question(test_question)
        
        if 'error' not in analysis:
            print("  ✓ Question analysis works")
        else:
            print(f"  Analysis returned: {analysis['error']}")
        
        # Test statistics
        stats = evaluator.get_semantic_statistics()
        print(f"  ✓ Statistics tracking: {len(stats)} metrics")
        
        print("Evaluator tests passed")
        return True
        
    except Exception as e:
        print(f"Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_usage_examples():
    """Create usage examples"""
    
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Create __init__.py for examples
    (examples_dir / "__init__.py").touch()
    
    # Basic usage example
    basic_example = '''#!/usr/bin/env python3
"""Basic Semantic Layer Usage Example"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from semantic_layer import create_semantic_layer, SemanticEvaluator

def example_intent_analysis():
    """Example of semantic intent analysis"""
    
    print("Semantic Intent Analysis Example")
    print("=" * 40)
    
    semantic_layer = create_semantic_layer()
    
    questions = [
        "How many cars were made in 1980?",
        "What is the average horsepower by manufacturer?",
        "Show me the top 10 cars with highest MPG",
        "List all students enrolled in computer science"
    ]
    
    for question in questions:
        print(f"\\nQuestion: {question}")
        analysis = semantic_layer.analyze_query_intent(question)
        
        print(f"  Metrics found: {len(analysis['relevant_metrics'])}")
        print(f"  Dimensions found: {len(analysis['relevant_dimensions'])}")
        print(f"  Complexity: {semantic_layer._assess_complexity(analysis)}")

if __name__ == "__main__":
    example_intent_analysis()
'''
    
    basic_file = examples_dir / "basic_usage.py"
    with open(basic_file, 'w') as f:
        f.write(basic_example)
    
    print(f"  ✓ Basic usage example created: {basic_file}")
    return [str(basic_file)]

def create_integration_example():
    """Create integration example for utils/eval.py"""
    
    integration_code = '''
# Semantic Layer Integration Guide
# Add this to your utils/eval.py file

# At the top with other imports:
try:
    from semantic_layer import SemanticEvaluator
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Semantic layer not available")

# In your argument parser section:
parser.add_argument('--use_semantic', default=False, action='store_true',
                    help='Use semantic layer for enhanced SQL generation')

# Where you create the evaluator, replace:
# evaluator = Evaluator(...)
# With:
if args.use_semantic and SEMANTIC_AVAILABLE:
    evaluator = SemanticEvaluator(
        prompt_type=args.prompt_type,
        enable_debugging=args.enable_debugging,
        use_chromadb=args.use_chromadb,
        chromadb_config=chromadb_config
    )
    print("Using Semantic Enhanced Evaluator")
else:
    evaluator = Evaluator(
        prompt_type=args.prompt_type,
        enable_debugging=args.enable_debugging,
        use_chromadb=args.use_chromadb,
        chromadb_config=chromadb_config
    )

# At the end of evaluation, optionally show semantic stats:
if args.use_semantic and SEMANTIC_AVAILABLE and hasattr(evaluator, 'get_semantic_statistics'):
    semantic_stats = evaluator.get_semantic_statistics()
    print("\\nSemantic Enhancement Statistics:")
    for key, value in semantic_stats.items():
        print(f"  {key}: {value}")
'''
    
    # Save integration example
    current_dir = Path(__file__).parent
    integration_file = current_dir / "INTEGRATION_GUIDE.txt"
    
    with open(integration_file, 'w') as f:
        f.write("Semantic Layer Integration Guide\n")
        f.write("=" * 40 + "\n\n")
        f.write(integration_code)
    
    print(f"  ✓ Integration guide created: {integration_file}")
    return str(integration_file)

def main():
    """Main setup function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup and test semantic layer package')
    parser.add_argument('--test', action='store_true', help='Run functionality tests')
    parser.add_argument('--examples', action='store_true', help='Create usage examples')
    parser.add_argument('--integration', action='store_true', help='Create integration guide')
    parser.add_argument('--all', action='store_true', 
                       help='Run all setup steps')
    
    args = parser.parse_args()
    
    if args.all:
        args.test = args.examples = args.integration = True
    
    print("Semantic Layer Package Setup")
    print("=" * 35)
    
    # Check package structure
    if not check_package_structure():
        return False
    
    # Validate config
    if not validate_config():
        return False
    
    # Validate imports
    if not validate_imports():
        return False
    
    # Run tests if requested
    if args.test:
        print("\n" + "=" * 35)
        if not test_semantic_functionality():
            return False
        if not test_evaluator_integration():
            return False
    
    # Create examples if requested
    if args.examples:
        print("\n" + "=" * 35)
        example_files = create_usage_examples()
        print(f"Examples created: {len(example_files)} files")
    
    # Create integration guide if requested
    if args.integration:
        print("\n" + "=" * 35)
        integration_file = create_integration_example()
    
    print("\n" + "=" * 35)
    print("✅ Semantic layer package setup complete!")
    print("\nUsage:")
    print("  from semantic_layer import SemanticEvaluator, create_semantic_layer")
    print("  python semantic_layer/examples/basic_usage.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)