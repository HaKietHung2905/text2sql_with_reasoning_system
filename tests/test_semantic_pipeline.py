"""Tests for semantic pipeline"""

import pytest
from src.semantic.semantic_pipeline import SemanticPipeline


def test_pipeline_init():
    """Test initialization"""
    pipeline = SemanticPipeline({'enabled': False})
    assert not pipeline.enabled


def test_enhancement():
    """Test enhancement"""
    pipeline = SemanticPipeline({'enabled': True})
    result = pipeline.enhance_question("How many cars?", "car_1", {})
    
    assert 'enhanced_question' in result
    assert 'suggestions' in result
    assert isinstance(result['suggestions'], list)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
