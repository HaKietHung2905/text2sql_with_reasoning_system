"""
SQL query hardness/difficulty classification.
Categorizes queries as easy, medium, hard, or extra hard.
"""

from typing import Dict, Set, List

# Constants
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')

HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


def get_keywords(sql: Dict) -> Set[str]:
    """Extract keywords from SQL structure"""
    res = set()
    
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')
    
    # OR keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')
    
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    
    # NOT keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')
    
    # IN keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')
    
    # LIKE keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')
    
    return res


def get_nestedSQL(sql: Dict) -> List[Dict]:
    """Get nested SQL queries"""
    nested = []
    
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    
    return nested


def count_component1(sql: Dict) -> int:
    """Count component1 complexity factors"""
    count = 0
    
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:
        count += len(sql['from']['table_units']) - 1
    
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])
    
    return count


def count_component2(sql: Dict) -> int:
    """Count component2 complexity factors (nested queries)"""
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql: Dict) -> int:
    """Count other complexity factors"""
    from src.evaluation.metrics import count_agg
    
    count = 0
    
    # Number of aggregations
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                              [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    
    if agg_count > 1:
        count += 1
    
    # Number of select columns
    if len(sql['select'][1]) > 1:
        count += 1
    
    # Number of where conditions
    if len(sql['where']) > 1:
        count += 1
    
    # Number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1
    
    return count


def eval_hardness(sql: Dict) -> str:
    """
    Evaluate query hardness/difficulty
    
    Args:
        sql: Parsed SQL structure
        
    Returns:
        Hardness level: 'easy', 'medium', 'hard', or 'extra'
    """
    count_comp1 = count_component1(sql)
    count_comp2 = count_component2(sql)
    count_other = count_others(sql)
    
    if count_comp1 <= 1 and count_other == 0 and count_comp2 == 0:
        return "easy"
    elif ((count_other <= 2 and count_comp1 <= 1 and count_comp2 == 0) or
          (count_comp1 <= 2 and count_other < 2 and count_comp2 == 0)):
        return "medium"
    elif ((count_other > 2 and count_comp1 <= 2 and count_comp2 == 0) or
          (2 < count_comp1 <= 3 and count_other <= 2 and count_comp2 == 0) or
          (count_comp1 <= 1 and count_other == 0 and count_comp2 <= 1)):
        return "hard"
    else:
        return "extra"