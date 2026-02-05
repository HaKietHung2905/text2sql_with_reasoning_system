"""
Evaluation compatibility module.
Exposes evaluators from src.evaluation for backward compatibility.
"""
import sys
import os
from pathlib import Path

# Add project root to path to ensure imports work
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.evaluation.base_evaluator import BaseEvaluator
except ImportError as e:
    # If src.evaluation isn't found, try local import if running from root
    try:
        from src.evaluation.base_evaluator import BaseEvaluator
    except ImportError:
        print(f"DEBUG: Failed to import BaseEvaluator in utils/eval.py: {e}")
        pass

# Re-export utils for compatibility
from utils.eval_utils import (
    normalize_sql_for_evaluation, 
    extract_db_name_from_question, 
    clean_sql_string
)

__all__ = [
    'BaseEvaluator',
    'normalize_sql_for_evaluation', 
    'extract_db_name_from_question', 
    'clean_sql_string'
]
