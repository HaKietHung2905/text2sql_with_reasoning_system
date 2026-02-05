"""
Main evaluation coordinator.
Handles the complete evaluation pipeline.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.sql_schema import Schema, get_schema
from src.data.sql_parser import parse_sql as get_sql  # Alias for compatibility
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.hardness import eval_hardness
from src.evaluation.sql_rebuilder import (
    build_valid_col_units, rebuild_sql_col, rebuild_sql_val, clean_query
)
from src.evaluation.result_formatter import print_scores
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import execution evaluator
try:
    from src.evaluation.exec_evaluator import eval_exec_match
    EXEC_EVAL_AVAILABLE = True
except ImportError:
    EXEC_EVAL_AVAILABLE = False
    logger.warning("exec_eval not available - execution accuracy disabled")


def evaluate(
    gold: str,
    predict: Optional[str],
    db_dir: str,
    etype: str,
    kmaps: Dict,
    plug_value: bool = False,
    keep_distinct: bool = False,
    progress_bar_for_each_datapoint: bool = False,
    use_langchain: bool = False,
    questions_file: Optional[str] = None,
    prompt_type: str = "enhanced",
    enable_debugging: bool = False,
    use_chromadb: bool = False,
    chromadb_config: Optional[Dict] = None,
    use_semantic: bool = False
) -> Dict:
    """
    Main evaluation function
    
    Args:
        gold: Path to gold queries file
        predict: Path to predicted queries file
        db_dir: Directory containing databases
        etype: Evaluation type ('all', 'exec', 'match')
        kmaps: Foreign key mappings
        plug_value: Whether to plug gold values
        keep_distinct: Whether to keep DISTINCT
        progress_bar_for_each_datapoint: Show progress per datapoint
        use_langchain: Generate SQL with LangChain
        questions_file: Path to questions file
        prompt_type: Type of prompt template
        enable_debugging: Enable debug mode
        use_chromadb: Use ChromaDB retrieval
        chromadb_config: ChromaDB configuration
        use_semantic: Use semantic layer
        
    Returns:
        Dictionary with evaluation results
    """
    # Create evaluator
    evaluator = create_evaluator(
        prompt_type=prompt_type,
        enable_debugging=enable_debugging,
        use_chromadb=use_chromadb,
        chromadb_config=chromadb_config,
        use_semantic=use_semantic
    )
    
    # Load or generate predictions
    if use_langchain and questions_file:
        plist = generate_predictions(
            evaluator, questions_file, db_dir
        )
    else:
        plist = load_predictions(predict)
    
    # Load gold queries
    glist = load_gold_queries(gold)
    
    # Check lengths match
    if len(plist) != len(glist):
        raise ValueError(f"Number of predictions ({len(plist)}) != gold ({len(glist)})")
    
    # Initialize scores
    scores = initialize_scores()
    
    # Evaluate each session
    entries = []
    for i, (p, g) in enumerate(zip(plist, glist)):
        scores['joint_all']['count'] += 1
        turn_scores = {"exec": [], "exact": []}
        
        # Evaluate each turn in session
        for idx, (p_turn, g_turn) in enumerate(zip(p, g)):
            result = evaluate_turn(
                p_turn, g_turn, db_dir, etype, kmaps,
                evaluator, plug_value, keep_distinct,
                progress_bar_for_each_datapoint
            )
            
            if result is None:
                continue
            
            # Update scores
            update_scores(scores, result, turn_scores, idx)
            entries.append(result['entry'])
        
        # Update joint scores
        if all(v == 1 for v in turn_scores["exec"]):
            scores['joint_all']['exec'] += 1
        if all(v == 1 for v in turn_scores["exact"]):
            scores['joint_all']['exact'] += 1
    
    # Calculate final scores
    finalize_scores(scores, etype)
    
    # Print results
    include_turn_acc = len(glist) > 1
    print_scores(scores, etype, include_turn_acc=include_turn_acc)
    
    # Print statistics if available
    print_statistics(evaluator, use_chromadb, use_semantic)
    
    # Return summary
    if etype == 'match':
        return {'exact': scores['all']['exact']}
    elif etype == 'exec':
        return {'exec': scores['all']['exec']}
    else:
        return {
            'exact': scores['all']['exact'],
            'exec': scores['all']['exec']
        }


def generate_predictions(evaluator, questions_file: str, db_dir: str) -> List[List]:
    """Generate predictions from questions using LangChain"""
    logger.info("🤖 Generating SQL from questions...")
    
    with open(questions_file, 'r') as f:
        lines = f.readlines()
    
    predictions = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Parse question and database
        parts = parse_question_line(line)
        if not parts:
            logger.warning(f"Skipping line {line_num}: {line}")
            continue
        
        question, db_name = parts
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        
        # Generate SQL
        if hasattr(evaluator, 'generate_sql_from_question'):
            sql = evaluator.generate_sql_from_question(question, db_path)
        else:
            from src.generation.sql_generator import SQLGenerator
            generator = SQLGenerator()
            sql = generator.generate(question, db_path)
        
        predictions.append([sql])
        logger.info(f"Generated: {sql}")
    
    return [predictions]


def parse_question_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse question line to extract question and database name"""
    parts = None
    
    if '\t' in line:
        parts = line.split('\t')
    elif '  ' in line:
        parts = re.split(r'\s{2,}', line)
    elif '\\t' in line:
        parts = line.split('\\t')
    elif ' ' in line:
        parts = line.rsplit(' ', 1)
    
    if not parts or len(parts) < 2:
        return None
    
    question = parts[0].strip()
    db_name = parts[1].strip()
    
    return question, db_name


def load_predictions(predict_path: str) -> List[List]:
    """Load predictions from file"""
    if not predict_path:
        return []
    
    with open(predict_path) as f:
        plist = []
        pseq_one = []
        
        for line in f.readlines():
            if len(line.strip()) == 0:
                plist.append(pseq_one)
                pseq_one = []
            else:
                pseq_one.append(line.strip().split('\t'))
        
        if len(pseq_one) != 0:
            plist.append(pseq_one)
    
    return plist


def load_gold_queries(gold_path: str) -> List[List]:
    """Load gold queries from file"""
    with open(gold_path) as f:
        glist = []
        gseq_one = []
        
        for line in f.readlines():
            if len(line.strip()) == 0:
                glist.append(gseq_one)
                gseq_one = []
            else:
                lstrip = line.strip().split('\t')
                gseq_one.append(lstrip)
        
        if len(gseq_one) != 0:
            glist.append(gseq_one)
    
    return glist


def evaluate_turn(
    p_turn, g_turn, db_dir: str, etype: str, kmaps: Dict,
    evaluator, plug_value: bool, keep_distinct: bool,
    progress_bar: bool
) -> Optional[Dict]:
    """Evaluate a single turn"""
    p_str = p_turn[0] if len(p_turn) > 0 else ""
    g_str, db = g_turn
    db_name = db
    db_path = os.path.join(db_dir, db, f"{db}.sqlite")
    
    # Parse gold SQL
    try:
        schema = load_schema(db_path)
        g_sql = parse_sql(g_str, schema)
    except Exception as e:
        logger.error(f"Error parsing gold SQL: {e}")
        return None
    
    hardness = eval_hardness(g_sql)
    
    # Parse predicted SQL
    try:
        if isinstance(p_str, str):
            p_str = p_str.strip('`').replace('`', '')
        p_sql = parse_sql(p_str, schema)
    except Exception as e:
        logger.warning(f"Parse failed: {e}")
        p_sql = get_empty_sql()
    
    result = {
        'hardness': hardness,
        'exec_score': 0,
        'exact_score': 0,
        'partial_scores': {},
        'entry': {
            'predictSQL': p_str,
            'goldSQL': g_str,
            'hardness': hardness
        }
    }
    
    # Execution accuracy
    if etype in ["all", "exec"] and EXEC_EVAL_AVAILABLE:
        exec_score = eval_exec_match(
            db=db_path, p_str=p_str, g_str=g_str,
            plug_value=plug_value, keep_distinct=keep_distinct,
            progress_bar_for_each_datapoint=progress_bar
        )
        result['exec_score'] = 1 if exec_score else 0
    
    # Exact match accuracy
    if etype in ["all", "match"]:
        # Rebuild SQL structures
        kmap = kmaps[db_name]
        
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        p_sql = clean_query(p_sql)
        
        exact_score = evaluator.eval_exact_match(p_sql, g_sql)
        result['exact_score'] = exact_score
        result['partial_scores'] = evaluator.partial_scores
        result['entry']['exact'] = exact_score
        result['entry']['partial'] = evaluator.partial_scores
    
    return result


def get_empty_sql() -> Dict:
    """Get empty SQL structure"""
    return {
        "except": None,
        "from": {"conds": [], "table_units": []},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [False, []],
        "union": None,
        "where": []
    }


def initialize_scores() -> Dict:
    """Initialize score tracking structure"""
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all', 'joint_all']
    partial_types = [
        'select', 'select(no AGG)', 'where', 'where(no OP)',
        'group(no Having)', 'group', 'order', 'and/or', 'IUEN', 'keywords'
    ]
    
    scores = {}
    
    for turn in turns:
        scores[turn] = {'count': 0, 'exact': 0.0, 'exec': 0}
    
    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.0, 'exec': 0}
        for type_ in partial_types:
            scores[level]['partial'][type_] = {
                'acc': 0.0, 'rec': 0.0, 'f1': 0.0,
                'acc_count': 0, 'rec_count': 0
            }
    
    return scores


def update_scores(scores: Dict, result: Dict, turn_scores: Dict, idx: int):
    """Update scores with evaluation result"""
    hardness = result['hardness']
    turn_id = "turn " + ("> 4" if idx > 3 else str(idx + 1))
    
    # Update counts
    scores[turn_id]['count'] += 1
    scores[hardness]['count'] += 1
    scores['all']['count'] += 1
    
    # Update execution scores
    if result['exec_score']:
        scores[hardness]['exec'] += 1
        scores[turn_id]['exec'] += 1
        scores['all']['exec'] += 1
        turn_scores['exec'].append(1)
    else:
        turn_scores['exec'].append(0)
    
    # Update exact match scores
    if result.get('partial_scores'):
        exact_score = result['exact_score']
        partial_scores = result['partial_scores']
        
        scores[turn_id]['exact'] += exact_score
        scores[hardness]['exact'] += exact_score
        scores['all']['exact'] += exact_score
        turn_scores['exact'].append(exact_score)
        
        # Update partial scores
        for type_, score in partial_scores.items():
            if score['pred_total'] > 0:
                scores[hardness]['partial'][type_]['acc'] += score['acc']
                scores[hardness]['partial'][type_]['acc_count'] += 1
                scores['all']['partial'][type_]['acc'] += score['acc']
                scores['all']['partial'][type_]['acc_count'] += 1
            
            if score['label_total'] > 0:
                scores[hardness]['partial'][type_]['rec'] += score['rec']
                scores[hardness]['partial'][type_]['rec_count'] += 1
                scores['all']['partial'][type_]['rec'] += score['rec']
                scores['all']['partial'][type_]['rec_count'] += 1
            
            scores[hardness]['partial'][type_]['f1'] += score['f1']
            scores['all']['partial'][type_]['f1'] += score['f1']


def finalize_scores(scores: Dict, etype: str):
    """Calculate final averaged scores"""
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all', 'joint_all']
    partial_types = [
        'select', 'select(no AGG)', 'where', 'where(no OP)',
        'group(no Having)', 'group', 'order', 'and/or', 'IUEN', 'keywords'
    ]
    
    # Finalize turn scores
    for turn in turns:
        if scores[turn]['count'] == 0:
            continue
        
        if etype in ["all", "exec"]:
            scores[turn]['exec'] /= scores[turn]['count']
        
        if etype in ["all", "match"]:
            scores[turn]['exact'] /= scores[turn]['count']
    
    # Finalize level scores
    for level in levels:
        if scores[level]['count'] == 0:
            continue
        
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']
        
        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            
            # Finalize partial scores
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = (
                        scores[level]['partial'][type_]['acc'] /
                        scores[level]['partial'][type_]['acc_count']
                    )
                
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = (
                        scores[level]['partial'][type_]['rec'] /
                        scores[level]['partial'][type_]['rec_count']
                    )
                
                acc = scores[level]['partial'][type_]['acc']
                rec = scores[level]['partial'][type_]['rec']
                
                if acc == 0 and rec == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = (
                        2.0 * acc * rec / (rec + acc)
                    )


def print_statistics(evaluator, use_chromadb: bool, use_semantic: bool):
    """Print additional statistics"""
    # Generation statistics
    if hasattr(evaluator, 'get_generation_statistics'):
        stats = evaluator.get_generation_statistics()
        if stats:
            logger.info("\n📊 Generation Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
    
    # ChromaDB statistics
    if use_chromadb and hasattr(evaluator, 'get_retrieval_statistics'):
        stats = evaluator.get_retrieval_statistics()
        logger.info("\n📊 ChromaDB Retrieval Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    # Semantic layer statistics
    if use_semantic and hasattr(evaluator, 'get_semantic_statistics'):
        stats = evaluator.get_semantic_statistics()
        logger.info("\n📊 Semantic Layer Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")


def create_evaluator(
    prompt_type: str = "enhanced",
    enable_debugging: bool = False,
    use_chromadb: bool = False,
    chromadb_config: Optional[Dict] = None,
    use_semantic: bool = False
):
    """
    Create appropriate evaluator based on configuration
    
    Args:
        prompt_type: Type of prompting strategy
        enable_debugging: Enable debugging features
        use_chromadb: Use ChromaDB retrieval
        chromadb_config: ChromaDB configuration
        use_semantic: Use semantic layer
        
    Returns:
        Configured evaluator instance
    """
    # Try to import advanced evaluators
    try:
        if use_semantic:
            from semantic_layer import SemanticEvaluator
            logger.info("🎯 Using Semantic Enhanced Evaluator")
            return SemanticEvaluator(
                prompt_type=prompt_type,
                enable_debugging=enable_debugging,
                use_chromadb=use_chromadb,
                chromadb_config=chromadb_config
            )
        elif use_chromadb:
            from src.evaluation.chromadb_evaluator import ChromaDBEvaluator
            logger.info("🔍 Using ChromaDB Evaluator")
            return ChromaDBEvaluator(
                prompt_type=prompt_type,
                enable_debugging=enable_debugging,
                chromadb_config=chromadb_config
            )
    except ImportError as e:
        logger.warning(f"Advanced evaluator not available: {e}")
    
    # Fallback to base evaluator
    logger.info("Using Base Evaluator")
    return BaseEvaluator()