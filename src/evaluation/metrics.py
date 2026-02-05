"""
Core evaluation metrics for Spider Text-to-SQL.
Implements exact match, partial match, and component-wise scoring.
"""

from typing import Dict, List, Tuple, Any

# Constants
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')


def get_scores(count: int, pred_total: int, label_total: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score
    
    Args:
        count: Number of correct predictions
        pred_total: Total predictions
        label_total: Total labels
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if pred_total != label_total:
        return 0.0, 0.0, 0.0
    elif count == pred_total:
        return 1.0, 1.0, 1.0
    return 0.0, 0.0, 0.0


def has_agg(unit: Tuple) -> bool:
    """Check if unit has aggregation"""
    return unit[0] != AGG_OPS.index('none')


def count_agg(units: List[Tuple]) -> int:
    """Count number of aggregations in units"""
    return len([unit for unit in units if has_agg(unit)])


def eval_sel(pred: Dict, label: Dict) -> Tuple[int, int, int, int]:
    """
    Evaluate SELECT clause
    
    Returns:
        Tuple of (label_total, pred_total, count, count_wo_agg)
    """
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0
    
    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])
    
    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred: Dict, label: Dict) -> Tuple[int, int, int, int]:
    """
    Evaluate WHERE clause
    
    Returns:
        Tuple of (label_total, pred_total, count, count_wo_agg)
    """
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0
    
    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])
    
    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    """Evaluate GROUP BY clause"""
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    
    # Normalize column names
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    
    return label_total, pred_total, cnt


def eval_having(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    """Evaluate HAVING clause"""
    pred_total = label_total = cnt = 0
    
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1
    
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    
    if (pred_total == label_total == 1 and
        pred_cols == label_cols and
        pred['having'] == label['having']):
        cnt = 1
    
    return label_total, pred_total, cnt


def eval_order(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    """Evaluate ORDER BY clause"""
    pred_total = label_total = cnt = 0
    
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    
    if (len(label['orderBy']) > 0 and 
        pred['orderBy'] == label['orderBy'] and
        ((pred['limit'] is None and label['limit'] is None) or
         (pred['limit'] is not None and label['limit'] is not None))):
        cnt = 1
    
    return label_total, pred_total, cnt


def eval_and_or(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    """Evaluate AND/OR operators"""
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)
    
    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0


def eval_keywords(pred: Dict, label: Dict) -> Tuple[int, int, int]:
    """Evaluate SQL keywords"""
    from src.evaluation.hardness import get_keywords
    
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0
    
    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    
    return label_total, pred_total, cnt