"""
Base evaluator class with core evaluation functionality.
"""

import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

from src.evaluation.metrics import (
    eval_sel, eval_where, eval_group, eval_having,
    eval_order, eval_and_or, eval_keywords, get_scores
)
from src.evaluation.hardness import eval_hardness
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseEvaluator:
    """Base evaluator with exact and partial match evaluation"""
    
    def __init__(self):
        self.partial_scores = None
        self.langchain_generator = None
        
    def eval_exact_match(self, pred: Dict, label: Dict) -> int:
        """
        Evaluate exact match between predicted and gold SQL
        
        Args:
            pred: Predicted SQL structure
            label: Gold SQL structure
            
        Returns:
            1 if exact match, 0 otherwise
        """
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores
        
        # Check all components match
        for key, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        
        # Check table units match
        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return 1 if label_tables == pred_tables else 0
        
        return 1
    
    def eval_partial_match(self, pred: Dict, label: Dict) -> Dict[str, Dict]:
        """
        Evaluate partial match across all SQL components
        
        Args:
            pred: Predicted SQL structure
            label: Gold SQL structure
            
        Returns:
            Dictionary with scores for each component
        """
        res = {}
        
        # SELECT clause
        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # WHERE clause
        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # GROUP BY clause
        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # HAVING clause
        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # ORDER BY clause
        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # AND/OR operators
        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # INTERSECT/UNION/EXCEPT
        label_total, pred_total, cnt = self._eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        # Keywords
        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {
            'acc': acc, 'rec': rec, 'f1': f1,
            'label_total': label_total, 'pred_total': pred_total
        }
        
        return res
    
    def _eval_IUEN(self, pred: Dict, label: Dict) -> Tuple[int, int, int]:
        """Evaluate INTERSECT, UNION, EXCEPT operations"""
        lt1, pt1, cnt1 = self._eval_nested(pred['intersect'], label['intersect'])
        lt2, pt2, cnt2 = self._eval_nested(pred['except'], label['except'])
        lt3, pt3, cnt3 = self._eval_nested(pred['union'], label['union'])
        
        label_total = lt1 + lt2 + lt3
        pred_total = pt1 + pt2 + pt3
        cnt = cnt1 + cnt2 + cnt3
        
        return label_total, pred_total, cnt
    
    def _eval_nested(self, pred: Optional[Dict], label: Optional[Dict]) -> Tuple[int, int, int]:
        """Evaluate nested SQL"""
        label_total = 0
        pred_total = 0
        cnt = 0
        
        if pred is not None:
            pred_total += 1
        if label is not None:
            label_total += 1
        if pred is not None and label is not None:
            cnt += self.eval_exact_match(pred, label)
        
        return label_total, pred_total, cnt
    
    def get_generation_statistics(self) -> Dict:
        """Get generation statistics"""
        return {}


def eval_nested(pred: Optional[Dict], label: Optional[Dict], evaluator) -> Tuple[int, int, int]:
    """Standalone function for evaluating nested SQL"""
    label_total = 0
    pred_total = 0
    cnt = 0
    
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += evaluator.eval_exact_match(pred, label)
    
    return label_total, pred_total, cnt