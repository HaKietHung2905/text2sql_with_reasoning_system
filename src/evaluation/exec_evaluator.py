"""
Execution-based evaluation for SQL queries.
Tests whether predicted and gold queries produce the same results.
"""

import os
import re
import asyncio
import sqlite3
import itertools
from typing import List, Tuple, Set, Iterator, Dict, Any
from collections import defaultdict, namedtuple
from itertools import product, chain
import random

try:
    import sqlparse
    from sqlparse.tokens import Whitespace
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Constants
Token = namedtuple('Token', ['ttype', 'value'])
VALUE_NUM_SYMBOL = 'VALUERARE'
QUOTE_CHARS = {'`', '\'', '"'}
TIMEOUT = 60


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    """Permute tuple elements according to permutation"""
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def tokenize(query: str) -> List:
    """Tokenize SQL query using sqlparse"""
    if not SQLPARSE_AVAILABLE:
        raise ImportError("sqlparse is required for tokenization")
    
    tokens = list([Token(t.ttype, t.value) for t in sqlparse.parse(query)[0].flatten()])
    return tokens


def strip_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Strip values from query and replace with placeholder
    
    Args:
        query: SQL query string
        
    Returns:
        Tuple of (query_keywords, all_values)
    """
    if not SQLPARSE_AVAILABLE:
        raise ImportError("sqlparse is required")
    
    query_keywords, all_values = [], []
    
    # Extract string literals
    toks = sqlparse.parse(query)[0].flatten()
    values = [
        t.value for t in toks 
        if t.ttype == sqlparse.tokens.Literal.String.Single or 
           t.ttype == sqlparse.tokens.Literal.String.Symbol
    ]
    
    # Replace string values with placeholder
    for val in values:
        all_values.append(val)
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)
    
    query_tokenized = query.split()
    
    # Extract float numbers
    float_nums = re.findall(r"[-+]?\d*\.\d+", query)
    all_values += [qt for qt in query_tokenized if qt in float_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]
    
    query = " ".join(query_tokenized)
    
    # Extract integer numbers
    int_nums = [i.strip() for i in re.findall(r"[^tT]\d+", query)]
    all_values += [qt for qt in query_tokenized if qt in int_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    
    # Process tokens
    for tok in query_tokenized:
        if "." in tok:
            table = re.findall(r"[Tt]\d+\.", tok)
            if len(table) > 0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t) > 0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())
        elif len(tok) > 0:
            query_keywords.append(tok.lower())
    
    return query_keywords, all_values


def reformat_query(query: str) -> str:
    """
    Reformat query to standardized form
    
    Args:
        query: SQL query string
        
    Returns:
        Reformatted query
    """
    if not SQLPARSE_AVAILABLE:
        raise ImportError("sqlparse is required")
    
    query = query.strip().replace(";", "").replace("\t", "")
    query = ' '.join([
        t.value for t in tokenize(query) 
        if t.ttype != sqlparse.tokens.Whitespace
    ])
    
    # Replace table aliases with wildcards
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    
    return query


def replace_values(sql: str) -> Tuple[List[str], Set[str]]:
    """
    Replace values in SQL with placeholders
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (query_tokens_no_value, values_set)
    """
    if not SQLPARSE_AVAILABLE:
        raise ImportError("sqlparse is required")
    
    sql = sqlparse.format(sql, reindent=False, keyword_case='upper')
    sql = re.sub(r"(T\d+\.)\s", r"\1", sql)
    query_toks_no_value, values = strip_query(sql)
    
    return query_toks_no_value, set(values)


def plugin(query_value_replaced: List[str], values_in_order: List[str]) -> str:
    """
    Plug values into query with value slots
    
    Args:
        query_value_replaced: Query with value placeholders
        values_in_order: Values to plug in
        
    Returns:
        Query with values plugged in
    """
    q_length = len(query_value_replaced)
    query_w_values = query_value_replaced[:]
    value_idx = [
        idx for idx in range(q_length) 
        if query_value_replaced[idx] == VALUE_NUM_SYMBOL.lower()
    ]
    
    assert len(value_idx) == len(values_in_order), \
        f"Mismatch: {len(value_idx)} slots but {len(values_in_order)} values"
    
    for idx, value in zip(value_idx, values_in_order):
        query_w_values[idx] = value
    
    return ' '.join(query_w_values)


def extract_query_values(sql: str) -> Tuple[List[str], Set[str]]:
    """
    Extract non-value tokens and set of values from SQL query
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (query_value_replaced, values_set)
    """
    reformatted = reformat_query(query=sql)
    query_value_replaced, values = replace_values(reformatted)
    return query_value_replaced, values


def plugin_all_permutations(query_value_replaced: List[str], values: Set[str]) -> Iterator[str]:
    """
    Generate all possible ways of filling values into query
    
    Args:
        query_value_replaced: Query with value placeholders
        values: Set of possible values
        
    Yields:
        Query with values plugged in
    """
    num_slots = len([v for v in query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    
    for value_combo in itertools.product(*[list(values) for _ in range(num_slots)]):
        yield plugin(query_value_replaced, list(value_combo))


def get_all_preds_for_execution(gold: str, pred: str) -> Tuple[int, Iterator[str]]:
    """
    Get all possible predictions by plugging gold values into predicted query
    
    Args:
        gold: Gold SQL query
        pred: Predicted SQL query
        
    Returns:
        Tuple of (num_alternatives, predictions_iterator)
    """
    _, gold_values = extract_query_values(gold)
    pred_query_value_replaced, _ = extract_query_values(pred)
    
    num_slots = len([v for v in pred_query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    num_alternatives = len(gold_values) ** num_slots
    
    return num_alternatives, plugin_all_permutations(pred_query_value_replaced, gold_values)


def remove_distinct(s: str) -> str:
    """Remove DISTINCT keyword from query"""
    if not SQLPARSE_AVAILABLE:
        return s.replace('DISTINCT', '').replace('distinct', '')
    
    toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
    return ''.join([t for t in toks if t.lower() != 'distinct'])


def postprocess(query: str) -> str:
    """Postprocess query to avoid execution errors"""
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return query


def replace_cur_year(query: str) -> str:
    """Replace CURDATE function with fixed year"""
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", 
        "2020", 
        query, 
        flags=re.IGNORECASE
    )


def get_cursor_from_path(sqlite_path: str):
    """
    Get database cursor for SQLite database
    
    Args:
        sqlite_path: Path to SQLite database
        
    Returns:
        Database cursor
    """
    try:
        if not os.path.exists(sqlite_path):
            logger.warning(f"Opening new connection: {sqlite_path}")
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        logger.error(f"Failed to connect to {sqlite_path}: {e}")
        raise e
    
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


async def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    """
    Execute query on database (internal async function)
    
    Args:
        sqlite_path: Path to SQLite database
        query: SQL query to execute
        
    Returns:
        Tuple of (status, result_or_exception)
    """
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e


async def exec_on_db(
    sqlite_path: str, 
    query: str, 
    process_id: str = "", 
    timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    """
    Execute query on database with timeout
    
    Args:
        sqlite_path: Path to SQLite database
        query: SQL query to execute
        process_id: Process identifier (unused, kept for compatibility)
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (status, result_or_exception)
    """
    try:
        return await asyncio.wait_for(exec_on_db_(sqlite_path, query), timeout)
    except asyncio.TimeoutError:
        return ('exception', TimeoutError)
    except Exception as e:
        return ("exception", e)


def unorder_row(row: Tuple) -> Tuple:
    """Sort row elements for comparison"""
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    Quick rejection test for result equivalence
    
    Args:
        result1: First result set
        result2: Second result set
        order_matters: Whether row order matters
        
    Returns:
        True if results might be equivalent, False if definitely not
    """
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    """
    Get constrained permutations for column matching
    
    Args:
        tab1_sets_by_columns: Sets of values in each column of table 1
        result2: Second result set
        
    Returns:
        Iterator of permutations
    """
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    
    if num_cols <= 3:
        return product(*perm_constraints)
    
    # Sample 20 rows to constrain permutation space
    for _ in range(20):
        random_tab2_row = random.choice(result2)
        
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    
    return product(*perm_constraints)


def multiset_eq(l1: List, l2: List) -> bool:
    """
    Check if two lists are equivalent as multisets
    
    Args:
        l1: First list
        l2: Second list
        
    Returns:
        True if equivalent as multisets
    """
    if len(l1) != len(l2):
        return False
    
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    
    return True


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    Check whether two result sets are equivalent
    
    Args:
        result1: First result set
        result2: Second result set
        order_matters: Whether row order matters
        
    Returns:
        True if results are equivalent
    """
    # Both empty
    if len(result1) == 0 and len(result2) == 0:
        return True
    
    # Different lengths
    if len(result1) != len(result2):
        return False
    
    # Different number of columns
    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False
    
    # Quick rejection test
    if not quick_rej(result1, result2, order_matters):
        return False
    
    # Find column and row permutations
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]
    
    # Try all column permutations
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    
    return False


def eval_exec_match(
    db: str, 
    p_str: str, 
    g_str: str, 
    plug_value: bool, 
    keep_distinct: bool, 
    progress_bar_for_each_datapoint: bool
) -> int:
    """
    Evaluate execution match between predicted and gold queries
    
    Args:
        db: Database file path
        p_str: Predicted SQL query
        g_str: Gold SQL query
        plug_value: Whether to plug gold values into prediction
        keep_distinct: Whether to keep DISTINCT keyword
        progress_bar_for_each_datapoint: Show progress bar
        
    Returns:
        1 if match, 0 otherwise
    """
    # Postprocess queries
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)
    
    order_matters = 'order by' in g_str.lower()
    
    # Find all databases in directory
    db_dir = os.path.dirname(db)
    db_paths = [
        os.path.join(db_dir, basename) 
        for basename in os.listdir(db_dir) 
        if '.sqlite' in basename
    ]
    
    preds = [p_str]
    
    # Plug in values if requested
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        preds = chain([p_str], preds)
    
    # Try each prediction
    for pred in preds:
        pred_passes = 1
        
        # Setup progress bar
        if progress_bar_for_each_datapoint and TQDM_AVAILABLE:
            ranger = tqdm.tqdm(db_paths, desc="Testing databases")
        else:
            ranger = db_paths
        
        # Test on each database
        for db_path in ranger:
            g_flag, g_denotation = asyncio.run(exec_on_db(db_path, g_str))
            p_flag, p_denotation = asyncio.run(exec_on_db(db_path, pred))
            
            # Gold query should execute successfully
            if g_flag == 'exception':
                logger.warning(f"Gold query failed on {db_path}")
                return 0
            
            # Prediction execution failed
            if p_flag == 'exception':
                pred_passes = 0
            
            # Results not equivalent
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            
            if pred_passes == 0:
                break
        
        # Prediction passed all databases
        if pred_passes == 1:
            return 1
    
    # None of the predictions passed
    return 0