#!/usr/bin/env python3
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
        print(f"\nQuestion: {question}")
        analysis = semantic_layer.analyze_query_intent(question)
        
        print(f"  Metrics found: {len(analysis['relevant_metrics'])}")
        print(f"  Dimensions found: {len(analysis['relevant_dimensions'])}")
        print(f"  Complexity: {semantic_layer._assess_complexity(analysis)}")

if __name__ == "__main__":
    example_intent_analysis()
