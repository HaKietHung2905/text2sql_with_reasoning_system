"""
SQL parser for Spider dataset.
Parses SQL queries into structured representations.

Structure:
- val: number(float)/string(str)/sql(dict)
- col_unit: (agg_id, col_id, isDistinct(bool))
- val_unit: (unit_op, col_unit1, col_unit2)
- table_unit: (table_type, col_unit/sql)
- cond_unit: (not_op, op_id, val_unit, val1, val2)
- condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
- sql: see SQL_STRUCTURE below
"""

from typing import Dict, List, Tuple, Any, Optional

from utils.sql_parser import (
    tokenize, skip_semicolon,
    CLAUSE_KEYWORDS, JOIN_KEYWORDS, WHERE_OPS, UNIT_OPS,
    AGG_OPS, COND_OPS, SQL_OPS, ORDER_OPS, TABLE_TYPE
)
from utils.sql_schema import Schema
from utils.logging_utils import get_logger

logger = get_logger(__name__)


"""
SQL Structure:
{
    'select': (isDistinct(bool), [(agg_id, val_unit), ...])
    'from': {'table_units': [table_unit1, ...], 'conds': condition}
    'where': condition
    'groupBy': [col_unit1, ...]
    'orderBy': ('asc'/'desc', [val_unit1, ...])
    'having': condition
    'limit': None/limit value
    'intersect': None/sql
    'except': None/sql
    'union': None/sql
}
"""


class SQLParser:
    """Parse SQL queries into structured format"""
    
    def __init__(self, schema: Schema):
        """
        Initialize parser
        
        Args:
            schema: Schema object
        """
        self.schema = schema
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse SQL query
        
        Args:
            query: SQL query string
            
        Returns:
            Parsed SQL structure
        """
        toks = tokenize(query)
        tables_with_alias = self.schema.get_tables_with_alias(toks)
        _, sql = self._parse_sql(toks, 0, tables_with_alias)
        return sql
    
    def _parse_col(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: Optional[List[str]] = None
    ) -> Tuple[int, str]:
        """Parse column reference"""
        tok = toks[start_idx]
        
        if tok == "*":
            return start_idx + 1, self.schema.idMap[tok]
        
        if '.' in tok:
            alias, col = tok.split('.')
            key = tables_with_alias[alias] + "." + col
            return start_idx + 1, self.schema.idMap[key]
        
        assert default_tables is not None and len(default_tables) > 0
        
        for alias in default_tables:
            table = tables_with_alias[alias]
            if tok in self.schema.schema[table]:
                key = table + "." + tok
                return start_idx + 1, self.schema.idMap[key]
        
        raise ValueError(f"Error parsing column: {tok}")
    
    def _parse_col_unit(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: Optional[List[str]] = None
    ) -> Tuple[int, Tuple[int, str, bool]]:
        """Parse column unit with aggregation"""
        idx = start_idx
        len_ = len(toks)
        isBlock = False
        isDistinct = False
        
        if toks[idx] == '(':
            isBlock = True
            idx += 1
        
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
            assert idx < len_ and toks[idx] == '('
            idx += 1
            
            if toks[idx] == "distinct":
                idx += 1
                isDistinct = True
            
            idx, col_id = self._parse_col(toks, idx, tables_with_alias, default_tables)
            assert idx < len_ and toks[idx] == ')'
            idx += 1
            
            return idx, (agg_id, col_id, isDistinct)
        
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        
        agg_id = AGG_OPS.index("none")
        idx, col_id = self._parse_col(toks, idx, tables_with_alias, default_tables)
        
        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        
        return idx, (agg_id, col_id, isDistinct)
    
    def _parse_val_unit(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: Optional[List[str]] = None
    ) -> Tuple[int, Tuple]:
        """Parse value unit with operations"""
        idx = start_idx
        len_ = len(toks)
        isBlock = False
        
        if toks[idx] == '(':
            isBlock = True
            idx += 1
        
        col_unit1 = None
        col_unit2 = None
        unit_op = UNIT_OPS.index('none')
        
        idx, col_unit1 = self._parse_col_unit(toks, idx, tables_with_alias, default_tables)
        
        if idx < len_ and toks[idx] in UNIT_OPS:
            unit_op = UNIT_OPS.index(toks[idx])
            idx += 1
            idx, col_unit2 = self._parse_col_unit(toks, idx, tables_with_alias, default_tables)
        
        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        
        return idx, (unit_op, col_unit1, col_unit2)
    
    def _parse_table_unit(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str]
    ) -> Tuple[int, str, str]:
        """Parse table unit"""
        idx = start_idx
        len_ = len(toks)
        key = tables_with_alias[toks[idx]]
        
        if idx + 1 < len_ and toks[idx + 1] == "as":
            idx += 3
        else:
            idx += 1
        
        return idx, self.schema.idMap[key], key
    
    def _parse_value(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: Optional[List[str]] = None
    ) -> Tuple[int, Any]:
        """Parse value (number, string, or subquery)"""
        idx = start_idx
        len_ = len(toks)
        isBlock = False
        
        if toks[idx] == '(':
            isBlock = True
            idx += 1
        
        if toks[idx] == 'select':
            idx, val = self._parse_sql(toks, idx, tables_with_alias)
        elif "\"" in toks[idx]:
            val = toks[idx]
            idx += 1
        else:
            try:
                val = float(toks[idx])
                idx += 1
            except:
                end_idx = idx
                while (end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')' and
                       toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and
                       toks[end_idx] not in JOIN_KEYWORDS):
                    end_idx += 1
                
                idx, val = self._parse_col_unit(
                    toks[start_idx: end_idx], 0, tables_with_alias, default_tables
                )
                idx = end_idx
        
        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        
        return idx, val
    
    def _parse_condition(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: Optional[List[str]] = None
    ) -> Tuple[int, List]:
        """Parse WHERE/HAVING conditions"""
        idx = start_idx
        len_ = len(toks)
        conds = []
        
        while idx < len_:
            idx, val_unit = self._parse_val_unit(toks, idx, tables_with_alias, default_tables)
            not_op = False
            
            if toks[idx] == 'not':
                not_op = True
                idx += 1
            
            assert idx < len_ and toks[idx] in WHERE_OPS
            op_id = WHERE_OPS.index(toks[idx])
            idx += 1
            
            val1 = val2 = None
            
            if op_id == WHERE_OPS.index('between'):
                idx, val1 = self._parse_value(toks, idx, tables_with_alias, default_tables)
                assert toks[idx] == 'and'
                idx += 1
                idx, val2 = self._parse_value(toks, idx, tables_with_alias, default_tables)
            else:
                idx, val1 = self._parse_value(toks, idx, tables_with_alias, default_tables)
                val2 = None
            
            conds.append((not_op, op_id, val_unit, val1, val2))
            
            if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or
                             toks[idx] in JOIN_KEYWORDS):
                break
            
            if idx < len_ and toks[idx] in COND_OPS:
                conds.append(toks[idx])
                idx += 1
        
        return idx, conds
    
    def _parse_select(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: Optional[List[str]] = None
    ) -> Tuple[int, Tuple[bool, List]]:
        """Parse SELECT clause"""
        idx = start_idx
        len_ = len(toks)
        
        assert toks[idx] == 'select'
        idx += 1
        
        isDistinct = False
        if idx < len_ and toks[idx] == 'distinct':
            idx += 1
            isDistinct = True
        
        val_units = []
        
        while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
            agg_id = AGG_OPS.index("none")
            if toks[idx] in AGG_OPS:
                agg_id = AGG_OPS.index(toks[idx])
                idx += 1
            
            idx, val_unit = self._parse_val_unit(toks, idx, tables_with_alias, default_tables)
            val_units.append((agg_id, val_unit))
            
            if idx < len_ and toks[idx] == ',':
                idx += 1
        
        return idx, (isDistinct, val_units)
    
    def _parse_from(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str]
    ) -> Tuple[int, List, List, List[str]]:
        """Parse FROM clause"""
        assert 'from' in toks[start_idx:]
        
        len_ = len(toks)
        idx = toks.index('from', start_idx) + 1
        default_tables = []
        table_units = []
        conds = []
        
        while idx < len_:
            isBlock = False
            if toks[idx] == '(':
                isBlock = True
                idx += 1
            
            if toks[idx] == 'select':
                idx, sql = self._parse_sql(toks, idx, tables_with_alias)
                table_units.append((TABLE_TYPE['sql'], sql))
            else:
                if idx < len_ and toks[idx] == 'join':
                    idx += 1
                
                idx, table_unit, table_name = self._parse_table_unit(toks, idx, tables_with_alias)
                table_units.append((TABLE_TYPE['table_unit'], table_unit))
                default_tables.append(table_name)
            
            if idx < len_ and toks[idx] == "on":
                idx += 1
                idx, this_conds = self._parse_condition(toks, idx, tables_with_alias, default_tables)
                if len(conds) > 0:
                    conds.append('and')
                conds.extend(this_conds)
            
            if isBlock:
                assert toks[idx] == ')'
                idx += 1
            
            if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
                break
        
        return idx, table_units, conds, default_tables
    
    def _parse_where(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: List[str]
    ) -> Tuple[int, List]:
        """Parse WHERE clause"""
        idx = start_idx
        len_ = len(toks)
        
        if idx >= len_ or toks[idx] != 'where':
            return idx, []
        
        idx += 1
        idx, conds = self._parse_condition(toks, idx, tables_with_alias, default_tables)
        return idx, conds
    
    def _parse_group_by(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: List[str]
    ) -> Tuple[int, List]:
        """Parse GROUP BY clause"""
        idx = start_idx
        len_ = len(toks)
        col_units = []
        
        if idx >= len_ or toks[idx] != 'group':
            return idx, col_units
        
        idx += 1
        assert toks[idx] == 'by'
        idx += 1
        
        while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, col_unit = self._parse_col_unit(toks, idx, tables_with_alias, default_tables)
            col_units.append(col_unit)
            
            if idx < len_ and toks[idx] == ',':
                idx += 1
            else:
                break
        
        return idx, col_units
    
    def _parse_order_by(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: List[str]
    ) -> Tuple[int, Tuple[str, List]]:
        """Parse ORDER BY clause"""
        idx = start_idx
        len_ = len(toks)
        val_units = []
        order_type = 'asc'
        
        if idx >= len_ or toks[idx] != 'order':
            return idx, (order_type, val_units)
        
        idx += 1
        assert toks[idx] == 'by'
        idx += 1
        
        while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, val_unit = self._parse_val_unit(toks, idx, tables_with_alias, default_tables)
            val_units.append(val_unit)
            
            if idx < len_ and toks[idx] in ORDER_OPS:
                order_type = toks[idx]
                idx += 1
            
            if idx < len_ and toks[idx] == ',':
                idx += 1
            else:
                break
        
        return idx, (order_type, val_units)
    
    def _parse_having(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str],
        default_tables: List[str]
    ) -> Tuple[int, List]:
        """Parse HAVING clause"""
        idx = start_idx
        len_ = len(toks)
        
        if idx >= len_ or toks[idx] != 'having':
            return idx, []
        
        idx += 1
        idx, conds = self._parse_condition(toks, idx, tables_with_alias, default_tables)
        return idx, conds
    
    def _parse_limit(
        self,
        toks: List[str],
        start_idx: int
    ) -> Tuple[int, Optional[int]]:
        """Parse LIMIT clause"""
        idx = start_idx
        len_ = len(toks)
        
        if idx < len_ and toks[idx] == 'limit':
            idx += 2
            if type(toks[idx - 1]) != int:
                return idx, 1
            return idx, int(toks[idx - 1])
        
        return idx, None
    
    def _parse_sql(
        self,
        toks: List[str],
        start_idx: int,
        tables_with_alias: Dict[str, str]
    ) -> Tuple[int, Dict[str, Any]]:
        """Parse complete SQL query"""
        isBlock = False
        len_ = len(toks)
        idx = start_idx
        
        sql = {}
        
        if toks[idx] == '(':
            isBlock = True
            idx += 1
        
        # Parse FROM to get default tables
        from_end_idx, table_units, conds, default_tables = self._parse_from(
            toks, start_idx, tables_with_alias
        )
        sql['from'] = {'table_units': table_units, 'conds': conds}
        
        # Parse SELECT
        _, select_col_units = self._parse_select(toks, idx, tables_with_alias, default_tables)
        idx = from_end_idx
        sql['select'] = select_col_units
        
        # Parse WHERE
        idx, where_conds = self._parse_where(toks, idx, tables_with_alias, default_tables)
        sql['where'] = where_conds
        
        # Parse GROUP BY
        idx, group_col_units = self._parse_group_by(toks, idx, tables_with_alias, default_tables)
        sql['groupBy'] = group_col_units
        
        # Parse HAVING
        idx, having_conds = self._parse_having(toks, idx, tables_with_alias, default_tables)
        sql['having'] = having_conds
        
        # Parse ORDER BY
        idx, order_col_units = self._parse_order_by(toks, idx, tables_with_alias, default_tables)
        sql['orderBy'] = order_col_units
        
        # Parse LIMIT
        idx, limit_val = self._parse_limit(toks, idx)
        sql['limit'] = limit_val
        
        idx = skip_semicolon(toks, idx)
        
        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        
        idx = skip_semicolon(toks, idx)
        
        # Parse INTERSECT/UNION/EXCEPT
        for op in SQL_OPS:
            sql[op] = None
        
        if idx < len_ and toks[idx] in SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, IUE_sql = self._parse_sql(toks, idx, tables_with_alias)
            sql[sql_op] = IUE_sql
        
        return idx, sql


# Convenience functions
def parse_sql(query: str, schema: Schema) -> Dict[str, Any]:
    """
    Parse SQL query with given schema
    
    Args:
        query: SQL query string
        schema: Schema object
        
    Returns:
        Parsed SQL structure
    """
    parser = SQLParser(schema)
    return parser.parse(query)


def load_data(fpath: str) -> Any:
    """
    Load JSON data from file
    
    Args:
        fpath: File path
        
    Returns:
        Loaded data
    """
    with open(fpath) as f:
        return json.load(f)