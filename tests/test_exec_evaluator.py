"""
Tests for execution evaluator.
"""

import pytest
from src.evaluation.exec_evaluator import (
    permute_tuple, unorder_row, multiset_eq, result_eq,
    postprocess, remove_distinct, plugin
)


class TestTupleOps:
    """Test tuple operations"""
    
    def test_permute_tuple(self):
        """Test tuple permutation"""
        element = (1, 2, 3)
        perm = (2, 0, 1)
        result = permute_tuple(element, perm)
        assert result == (3, 1, 2)
    
    def test_unorder_row(self):
        """Test row unordering"""
        row = (3, 1, 2)
        result = unorder_row(row)
        # Should be sorted
        assert result == (1, 2, 3)


class TestMultiset:
    """Test multiset operations"""
    
    def test_multiset_eq_equal(self):
        """Test equal multisets"""
        l1 = [1, 2, 2, 3]
        l2 = [2, 1, 3, 2]
        assert multiset_eq(l1, l2)
    
    def test_multiset_eq_different(self):
        """Test different multisets"""
        l1 = [1, 2, 2, 3]
        l2 = [1, 2, 3, 3]
        assert not multiset_eq(l1, l2)
    
    def test_multiset_eq_different_length(self):
        """Test different length"""
        l1 = [1, 2, 3]
        l2 = [1, 2, 3, 4]
        assert not multiset_eq(l1, l2)


class TestResultEquivalence:
    """Test result equivalence checking"""
    
    def test_result_eq_empty(self):
        """Test both results empty"""
        assert result_eq([], [], False)
    
    def test_result_eq_different_length(self):
        """Test different lengths"""
        r1 = [(1, 2), (3, 4)]
        r2 = [(1, 2)]
        assert not result_eq(r1, r2, False)
    
    def test_result_eq_same_unordered(self):
        """Test same results unordered"""
        r1 = [(1, 2), (3, 4)]
        r2 = [(3, 4), (1, 2)]
        assert result_eq(r1, r2, False)
    
    def test_result_eq_same_ordered(self):
        """Test same results ordered"""
        r1 = [(1, 2), (3, 4)]
        r2 = [(1, 2), (3, 4)]
        assert result_eq(r1, r2, True)
    
    def test_result_eq_different_order(self):
        """Test different order when order matters"""
        r1 = [(1, 2), (3, 4)]
        r2 = [(3, 4), (1, 2)]
        assert not result_eq(r1, r2, True)


class TestQueryProcessing:
    """Test query processing"""
    
    def test_postprocess(self):
        """Test query postprocessing"""
        query = "SELECT * WHERE x > = 5 AND y < = 10"
        result = postprocess(query)
        assert result == "SELECT * WHERE x >= 5 AND y <= 10"
    
    def test_remove_distinct(self):
        """Test DISTINCT removal"""
        query = "SELECT DISTINCT name FROM users"
        result = remove_distinct(query)
        assert 'distinct' not in result.lower()
    
    def test_plugin(self):
        """Test value plugging"""
        query = ['select', '*', 'where', 'x', '=', 'valuerare']
        values = ['100']
        result = plugin(query, values)
        assert result == 'select * where x = 100'