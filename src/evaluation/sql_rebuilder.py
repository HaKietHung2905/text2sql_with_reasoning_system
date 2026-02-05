"""
SQL structure rebuilding for normalization.
Handles column mapping, value handling, and structure normalization.
"""

from typing import Dict, List, Any, Optional, Tuple

# Constants
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

# Flags
DISABLE_VALUE = True
DISABLE_DISTINCT = True


def build_valid_col_units(table_units: List, schema) -> List[str]:
    """Build list of valid column units from table units"""
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    prefixs = [col_id[:-2] for col_id in col_ids]
    
    valid_col_units = []
    for value in schema.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    
    return valid_col_units


def rebuild_col_unit_col(valid_col_units: List, col_unit: Optional[Tuple], kmap: Dict) -> Optional[Tuple]:
    """Rebuild column unit with column mapping"""
    if col_unit is None:
        return col_unit
    
    agg_id, col_id, distinct = col_unit
    
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    
    if DISABLE_DISTINCT:
        distinct = None
    
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units: List, val_unit: Optional[Tuple], kmap: Dict) -> Optional[Tuple]:
    """Rebuild value unit with column mapping"""
    if val_unit is None:
        return val_unit
    
    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units: List, table_unit: Optional[Tuple], kmap: Dict) -> Optional[Tuple]:
    """Rebuild table unit with column mapping"""
    if table_unit is None:
        return table_unit
    
    table_type, col_unit_or_sql = table_unit
    
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units: List, cond_unit: Optional[Tuple], kmap: Dict) -> Optional[Tuple]:
    """Rebuild condition unit with column mapping"""
    if cond_unit is None:
        return cond_unit
    
    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units: List, condition: List, kmap: Dict) -> List:
    """Rebuild condition list with column mapping"""
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition


def rebuild_from_col(valid_col_units: List, from_: Optional[Dict], kmap: Dict) -> Optional[Dict]:
    """Rebuild FROM clause with column mapping"""
    if from_ is None:
        return from_
    
    from_['table_units'] = [
        rebuild_table_unit_col(valid_col_units, table_unit, kmap) 
        for table_unit in from_['table_units']
    ]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    
    return from_


def rebuild_group_by_col(valid_col_units: List, group_by: Optional[List], kmap: Dict) -> Optional[List]:
    """Rebuild GROUP BY with column mapping"""
    if group_by is None:
        return group_by
    
    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units: List, order_by: Optional[Tuple], kmap: Dict) -> Optional[Tuple]:
    """Rebuild ORDER BY with column mapping"""
    if order_by is None or len(order_by) == 0:
        return order_by
    
    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    
    return direction, new_val_units


def rebuild_select_col(valid_col_units: List, sel: Optional[Tuple], kmap: Dict) -> Optional[Tuple]:
    """Rebuild SELECT clause with column mapping"""
    if sel is None:
        return sel
    
    distinct, _list = sel
    new_list = []
    
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    
    if DISABLE_DISTINCT:
        distinct = None
    
    return distinct, new_list


def rebuild_sql_col(valid_col_units: List, sql: Optional[Dict], kmap: Dict) -> Optional[Dict]:
    """Rebuild entire SQL structure with column mapping"""
    if sql is None:
        return sql
    
    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)
    
    return sql


def rebuild_cond_unit_val(cond_unit: Optional[Tuple]) -> Optional[Tuple]:
    """Rebuild condition unit values"""
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit
    
    not_op, op_id, val_unit, val1, val2 = cond_unit
    
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_val(condition: Optional[List]) -> Optional[List]:
    """Rebuild condition list values"""
    if condition is None or not DISABLE_VALUE:
        return condition
    
    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    
    return res


def rebuild_sql_val(sql: Optional[Dict]) -> Optional[Dict]:
    """Rebuild SQL structure values"""
    if sql is None or not DISABLE_VALUE:
        return sql
    
    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])
    
    return sql


def clean_query(sql_dict: Any) -> Any:
    """
    Recursively clean backticks from SQL query dictionary
    
    Args:
        sql_dict: SQL dictionary or value
        
    Returns:
        Cleaned SQL structure
    """
    if isinstance(sql_dict, dict):
        cleaned = {}
        for key, value in sql_dict.items():
            cleaned[key] = clean_query(value)
        return cleaned
    elif isinstance(sql_dict, list):
        return [clean_query(item) for item in sql_dict]
    elif isinstance(sql_dict, str):
        return sql_dict.strip('`').replace('`', '')
    else:
        return sql_dict