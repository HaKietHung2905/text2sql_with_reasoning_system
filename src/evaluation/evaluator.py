"""
Main evaluation coordinator.
Handles the complete evaluation pipeline.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.sql_schema import Schema, load_schema
from src.data.sql_parser import parse_sql
get_sql = parse_sql  # Alias for compatibility
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.hardness import eval_hardness
from src.evaluation.sql_rebuilder import (
    build_valid_col_units, rebuild_sql_col, rebuild_sql_val, clean_query
)
from src.evaluation.result_formatter import print_scores
from utils.logging_utils import get_logger
from utils.sql_schema import Schema, load_schema
from pathlib import Path
from utils.eval_utils import normalize_sql_for_evaluation


try:
    from src.semantic.semantic_pipeline import SemanticPipeline
    SEMANTIC_PIPELINE_AVAILABLE = True
except ImportError:
    SEMANTIC_PIPELINE_AVAILABLE = False
    logger.warning("Semantic pipeline not available")



logger = get_logger(__name__)

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
    use_semantic: bool = False,
    semantic_config: Optional[Dict] = None,
    limit: Optional[int] = None
) -> Dict:
    """Main evaluation function"""
    
    # Create evaluator (existing code)
    evaluator = create_evaluator(
        prompt_type=prompt_type,
        enable_debugging=enable_debugging,
        use_chromadb=use_chromadb,
        chromadb_config=chromadb_config,
        use_semantic=use_semantic
    )
    
    # Load gold queries
    glist = []
    if gold.endswith('.json'):
        with open(gold, 'r') as f:
            data = json.load(f)
            for item in data:
                glist.append([item['query'], item['db_id']])
    else:
        glist = load_gold_queries(gold)
    
    # Apply limit
    if limit:
        glist = glist[:limit]
        logger.info(f"Limiting evaluation to {limit} examples")
    
    # Load or generate predictions
    plist = []
    if use_langchain:
        # === SEMANTIC PIPELINE INIT ===
        semantic_pipeline = None
        if use_semantic and SEMANTIC_PIPELINE_AVAILABLE:
            try:
                semantic_pipeline = SemanticPipeline(semantic_config or {'enabled': True})
                logger.info("✓ Semantic pipeline enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic pipeline: {e}")

        # === GENERATION ===
        # Load examples
        examples = []
        if questions_file and questions_file.endswith('.json'):
            with open(questions_file, 'r') as f:
                examples = json.load(f)
        elif questions_file:
            with open(questions_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        examples.append({'question': parts[0], 'db_id': parts[1]})
        
        # Determine examples from gold if not provided (though logic above handles it via questions_file arg)
        if not examples and gold.endswith('.json'):
             # If questions_file was not set/valid but gold is json, we can use gold as examples source
             # strictly speaking evaluate_spider.py sets questions=gold if needed.
             pass

        # Check limit
        if hasattr(evaluator, 'limit') and evaluator.limit: # Actually arg passed to main, passed to evaluate?
             # args.limit is passed to evaluate as "limit" kwarg? NO. 
             # evaluate signature doesn't have limit.
             # I should check evaluate signature. 
             pass

        if limit:
             examples = examples[:limit]
             logger.info(f"Limiting examples to {limit}")

        schema_mapping = {}
        generator = None
        if not hasattr(evaluator, 'generate_sql_from_question'):
             try:
                 from src.generation.sql_generator import SQLGenerator
                 generator = SQLGenerator()
             except ImportError:
                 logger.warning("SQLGenerator not available for fallback")

        # Iterate
        # Note: simplistic assumption that examples order matches gold order if both from dev.json
        
        preds_list = [] # Temporary list for this loop
        questions_list = []

        loop_iter = examples
        # Handle limit if I add it to signature, but for now iterate all
        
        for i, example in enumerate(tqdm(loop_iter, desc="Generating SQL")):
            question = example.get('question', '')
            questions_list.append(question)
            db_id = example.get('db_id', '')
            if not question or not db_id:
                # Should we append empty? To keep alignment?
                # Yes, prediction must align with gold.
                preds_list.append("SELECT 1") 
                
                continue

            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            
            # Semantic Enhancement
            enhanced_question = question
            if semantic_pipeline:
                 try:
                     # Lazy load schema if needed, or pass None
                     res = semantic_pipeline.enhance_question(question, db_id, None)
                     enhanced_question = res['enhanced_question']
                     if res['enhanced']:
                         logger.debug(f"Enhanced: {question[:30]}... -> {len(res['suggestions'])} hints")
                 except Exception as e:
                     logger.warning(f"Enhancement error: {e}")
            
            # Generate
            generated_sql = "SELECT 1"
            try:
                if hasattr(evaluator, 'generate_sql_from_question'):
                    generated_sql = evaluator.generate_sql_from_question(enhanced_question, db_path)
                elif generator:
                    generated_sql = generator.generate(enhanced_question, db_path)
            except Exception as e:
                logger.error(f"Gen error: {e}")
            
            preds_list.append(generated_sql)
        
        # Convert to list of lists [ [sql] ] to match plist format
        plist = [[p] for p in preds_list]
        
    else:
        plist = load_predictions(predict)

    # Validate lengths
    if len(plist) != len(glist):
        logger.warning(f"Length mismatch: Preds={len(plist)}, Gold={len(glist)}. Truncating/Padding.")
        # Logic to handle mismatch?
        # Usually exact match required.
    
    # === EVALUATION ===
    scores = initialize_scores()
    
    # We need to know if we are in 'limit' mode? 
    # If we generated fewer preds, we should only eval those?
    # Or eval all and count missing as wrong?
    
    num_eval = min(len(plist), len(glist))
    
    
    # Save detailed results
    detailed_results = []
    
    for i in tqdm(range(num_eval), desc="Evaluating"):
        p_turn = plist[i]
        g_turn = glist[i]
        
        res = evaluate_turn(
            p_turn, g_turn, db_dir, etype, kmaps,
            evaluator, plug_value, keep_distinct,
            progress_bar_for_each_datapoint
        )
        
        if res:
            update_scores(scores, res, {'exec': [], 'exact': []}, i)
            
            # Add to detailed results
            detailed_results.append({
                'question': questions_list[i] if i < len(questions_list) else g_turn[0],
                # Actually g_turn is loaded from gold file. If JSON, it's [query, db]. If TSV, it's [sql, db].
                # Wait, load_gold_queries returns list of lists.
                # Let's assume prediction order matches gold order and we can get question from examples list if available
                # But examples list was created in the generation block loops.
                # simpler: just save SQLs.
                'gold_sql': res['entry']['goldSQL'],
                'predicted_sql': res['entry']['predictSQL'],
                'db_id': g_turn[1] if len(g_turn) > 1 else "Unknown",
                'exact_match': bool(res['exact_score']),
                'hardness': res['hardness']
            })
    
    # Export to file
    try:
        output_file = 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")

    finalize_scores(scores, etype)
    results = scores
    
    # Add stats
    results['exact_match_accuracy'] = scores['all']['exact']
    results['execution_accuracy'] = scores['all']['exec']

    if use_langchain and use_semantic and SEMANTIC_PIPELINE_AVAILABLE and semantic_pipeline:
         try:
             results['semantic_statistics'] = semantic_pipeline.get_statistics()
         except: pass

    return results

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
    p_turn, g_turn, db_dir, etype, kmaps,
    evaluator, plug_value, keep_distinct, progress_bar
) -> Optional[Dict]:
    """
    Evaluate a single turn with SQL normalization
    
    Args:
        p_turn: Predicted turn [sql, db]
        g_turn: Gold turn [sql, db]
        db_dir: Database directory
        etype: Evaluation type ('match', 'exec', 'all')
        kmaps: Key mappings
        evaluator: Evaluator instance
        plug_value: Whether to plug values
        keep_distinct: Whether to keep DISTINCT
        progress_bar: Whether to show progress bar
        
    Returns:
        Evaluation result dictionary or None if error
    """
    # Extract SQL strings
    p_str, g_str = p_turn[0], g_turn[0]
    db_name = g_turn[1]
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    
    # ========== NORMALIZATION ADDED HERE ==========
    # Normalize predicted SQL BEFORE parsing
    # This handles newlines, spacing, and case issues
    if isinstance(p_str, str):
        # First normalize for Spider format
        p_str = normalize_sql_for_evaluation(p_str)
        # Then clean backticks as before
        p_str = p_str.strip('`').replace('`', '')
    
    # Also normalize gold SQL for consistency
    if isinstance(g_str, str):
        g_str = normalize_sql_for_evaluation(g_str)
    # ==============================================
    
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
        p_sql = parse_sql(p_str, schema)
    except Exception as e:
        logger.warning(f"Parse failed: {e}")
        logger.debug(f"Failed SQL: {p_str}")
        p_sql = get_empty_sql()
    
    # Rest of the function continues as normal...
    result = {
        'hardness': hardness,
        'exec_score': 0,
        'exact_score': 0,
        'partial_scores': {},
        'entry': {
            'predictSQL': p_str,  # This now contains normalized SQL
            'goldSQL': g_str,      # This now contains normalized SQL
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
    use_semantic: bool = False  # ADD THIS PARAMETER
):
    """Create appropriate evaluator based on configuration"""
    
    # NOTE: We keep the old SemanticEvaluator path for backwards compatibility,
    # but the NEW semantic pipeline is separate and better
    
    try:
        if use_semantic:
            # Still import SemanticEvaluator for backwards compatibility
            from semantic_layer import SemanticEvaluator
            logger.info("🎯 Using Semantic Enhanced Evaluator (legacy)")
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