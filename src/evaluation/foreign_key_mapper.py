"""
Foreign key mapping utilities.
"""

import json
from typing import Dict, List, Set
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_foreign_key_map(entry: Dict) -> Dict[str, str]:
    """
    Build foreign key mapping for a database
    
    Args:
        entry: Database schema entry
        
    Returns:
        Dictionary mapping column IDs to their canonical forms
    """
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]
    
    # Rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if isinstance(col_orig, list) and len(col_orig) == 2:
            table_idx, col_name = col_orig
            if isinstance(table_idx, int) and table_idx >= 0:
                t = tables_orig[table_idx]
                c = col_name
                cols.append(f"__{t.lower()}.{c.lower()}__")
            else:
                cols.append("__all__")
        elif isinstance(col_orig, list) and len(col_orig) == 1:
            cols.append("__all__")
        else:
            cols.append("__all__")
    
    def keyset_in_list(k1: int, k2: int, k_list: List[Set]) -> Set:
        """Find or create keyset containing k1 or k2"""
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set
    
    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)
    
    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]
    
    return foreign_key_map


def build_foreign_key_map_from_json(table_path: str) -> Dict[str, Dict]:
    """
    Build foreign key maps for all databases from tables.json
    
    Args:
        table_path: Path to tables.json
        
    Returns:
        Dictionary mapping db_id to foreign key maps
    """
    try:
        with open(table_path) as f:
            data = json.load(f)
        
        tables = {}
        for entry in data:
            tables[entry['db_id']] = build_foreign_key_map(entry)
        
        logger.info(f"Built foreign key maps for {len(tables)} databases")
        return tables
        
    except Exception as e:
        logger.error(f"Failed to build foreign key maps: {e}")
        return {}