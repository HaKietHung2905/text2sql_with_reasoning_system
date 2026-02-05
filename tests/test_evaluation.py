"""
Tests for evaluation components.
"""

import pytest
from src.evaluation.metrics import (
    get_scores, eval_sel, eval_where, eval_group,
    eval_having, eval_order, eval_and_or
)
from src.evaluation.hardness import eval_hardness, count_component1, count_component2
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.sql_rebuilder import clean_query


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_get_scores_perfect(self):
        """Test perfect score"""
        acc, rec, f1 = get_scores(5, 5, 5)
        assert acc == 1.0
        assert rec == 1.0
        assert f1 == 1.0
    
    def test_get_scores_mismatch(self):
        """Test mismatched totals"""
        acc, rec, f1 = get_scores(3, 5, 4)
        assert acc == 0.0
        assert rec == 0.0
        assert f1 == 0.0
    
    def test_get_scores_zero(self):
        """Test zero score"""
        acc, rec, f1 = get_scores(0, 5, 5)
        assert acc == 0.0
        assert rec == 0.0
        assert f1 == 0.0


class TestHardness:
    """Test hardness evaluation"""
    
    def test_easy_query(self):
        """Test easy query classification"""
        sql = {
            'select': [False, [(0, (0, (0, 'col1', False), None))]],
            'from': {'table_units': [('table_unit', 'table1')], 'conds': []},
            'where': [],
            'groupBy': [],
            'having': [],
            'orderBy': [],
            'limit': None,
            'intersect': None,
            'except': None,
            'union': None
        }
        
        hardness = eval_hardness(sql)
        assert hardness == "easy"
    
    def test_count_component1(self):
        """Test component1 counting"""
        sql = {
            'where': [(False, 1, (0, (0, 'col1', False), None), 'value', None)],  # Proper condition unit
            'groupBy': [(0, 'col1', False)],  # Proper column unit
            'orderBy': ['asc', [(0, (0, 'col1', False), None)]],  # Proper order by
            'limit': 10,
            'from': {'table_units': [('table_unit', 'table1'), ('table_unit', 'table2')], 'conds': []},
            'having': []
        }
        count = count_component1(sql)
        assert count >= 4  # where, group, order, limit, join


class TestBaseEvaluator:
    """Test base evaluator"""
    
    def setup_method(self):
        """Setup evaluator"""
        self.evaluator = BaseEvaluator()
    
    def test_eval_exact_match_identical(self):
        """Test exact match with identical queries"""
        sql = {
            'select': [False, [(0, (0, (0, 'col1', False), None))]],
            'from': {'table_units': [('table_unit', 'table1')], 'conds': []},
            'where': [],
            'groupBy': [],
            'having': [],
            'orderBy': [],
            'limit': None,
            'intersect': None,
            'except': None,
            'union': None
        }
        
        result = self.evaluator.eval_exact_match(sql, sql)
        assert result == 1
    
    def test_eval_partial_match(self):
        """Test partial match evaluation"""
        sql1 = {
            'select': [False, [(0, (0, (0, 'col1', False), None))]],
            'from': {'table_units': [('table_unit', 'table1')], 'conds': []},
            'where': [],
            'groupBy': [],
            'having': [],
            'orderBy': [],
            'limit': None,
            'intersect': None,
            'except': None,
            'union': None
        }
        
        scores = self.evaluator.eval_partial_match(sql1, sql1)
        
        assert 'select' in scores
        assert 'where' in scores
        assert scores['select']['f1'] == 1.0


class TestSQLRebuilder:
    """Test SQL rebuilder"""
    
    def test_clean_query_dict(self):
        """Test cleaning query dictionary"""
        sql_dict = {
            'select': [False, ['`col1`', '`col2`']],
            'from': {'table_units': ['`table1`']},
            'where': []
        }
        
        cleaned = clean_query(sql_dict)
        
        # Check actual values are cleaned
        assert cleaned['select'][1] == ['col1', 'col2']
        assert cleaned['from']['table_units'] == ['table1']
    
    def test_clean_query_string(self):
        """Test cleaning string"""
        result = clean_query('`column_name`')
        assert result == 'column_name'
    
    def test_clean_query_list(self):
        """Test cleaning list"""
        result = clean_query(['`col1`', '`col2`'])
        assert result == ['col1', 'col2']


class TestEvalUtils:
    """Test evaluation utilities"""
    
    def test_normalize_sql(self):
        """Test SQL normalization"""
        from utils.eval_utils import normalize_sql_for_evaluation
        
        sql = "SELECT  * \n FROM   table1  ;"
        normalized = normalize_sql_for_evaluation(sql)
        
        assert normalized == "SELECT * FROM table1"
    
    def test_extract_db_name(self):
        """Test database name extraction"""
        from utils.eval_utils import extract_db_name_from_question
        
        line = "What is the total count?\tconcert_singer"
        result = extract_db_name_from_question(line)
        
        assert result is not None
        question, db_name = result
        assert question == "What is the total count?"
        assert db_name == "concert_singer"