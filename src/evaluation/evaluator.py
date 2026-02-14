"""
Main evaluation coordinator with Spider JSON support.
Handles the complete evaluation pipeline for Spider dataset.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from utils.sql_schema import Schema, get_schema
from src.data.sql_parser import parse_sql
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.hardness import eval_hardness
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

# Try to import semantic pipeline
try:
    from semantic_layer.core import SemanticPipeline
    SEMANTIC_PIPELINE_AVAILABLE = True
except ImportError:
    SEMANTIC_PIPELINE_AVAILABLE = False
    logger.warning("Semantic layer not available")


def load_schema(db_path: str):
    """Load database schema"""
    return Schema(get_schema(db_path))


def get_empty_sql():
    """Return empty SQL structure"""
    return {
        'select': [False, []],
        'from': {'table_units': [], 'conds': []},
        'where': [],
        'groupBy': [],
        'having': [],
        'orderBy': [],
        'limit': None,
        'intersect': None,
        'union': None,
        'except': None
    }


def normalize_sql_for_evaluation(sql: str) -> str:
    """
    Normalize SQL for Spider evaluation.
    Handles newlines, whitespace, and formatting differences.
    """
    if not sql or not sql.strip():
        return sql
    
    # Remove newlines and collapse whitespace
    sql = re.sub(r'\s+', ' ', sql)
    
    # Lowercase keywords
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 
        'ORDER', 'LIMIT', 'JOIN', 'ON', 'AND', 'OR', 'IN',
        'NOT', 'LIKE', 'BETWEEN', 'DISTINCT', 'AS', 'UNION',
        'INTERSECT', 'EXCEPT', 'LEFT', 'RIGHT', 'INNER', 'OUTER'
    ]
    for kw in keywords:
        sql = re.sub(rf'\b{kw}\b', kw.lower(), sql, flags=re.IGNORECASE)
    
    # Uppercase aggregate functions (Spider convention)
    functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    for fn in functions:
        sql = re.sub(rf'\b{fn}\b', fn, sql, flags=re.IGNORECASE)
    
    # Normalize punctuation spacing
    sql = re.sub(r'\s*,\s*', ' , ', sql)
    sql = re.sub(r'\s*\(\s*', ' ( ', sql)
    sql = re.sub(r'\s*\)\s*', ' ) ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    
    # Final cleanup
    sql = re.sub(r'\s+', ' ', sql)
    
    return sql.strip()


def load_gold_from_spider_json(json_path: str, limit: Optional[int] = None) -> List[List]:
    """
    Load gold queries from Spider JSON format
    
    Spider JSON format:
    [
        {
            "db_id": "concert_singer",
            "question": "How many singers do we have?",
            "query": "SELECT count(*) FROM singer"
        },
        ...
    ]
    
    Args:
        json_path: Path to Spider JSON file (dev.json, train.json)
        limit: Optional limit on number of examples
        
    Returns:
        List of lists in format [[[sql, db_id]], [[sql, db_id]], ...]
        (wrapped in double list for single-turn interactions)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        logger.info(f"Limited to {limit} examples")
    
    # Convert to format expected by evaluation pipeline
    # Each entry becomes a list with single turn: [[sql, db_id]]
    glist = []
    for item in data:
        query = item.get('query', item.get('sql', ''))
        db_id = item.get('db_id', '')
        if not query or not db_id:
            logger.warning(f"Skipping invalid item: {item}")
            continue
        # Wrap in list to represent single-turn interaction
        glist.append([[query, db_id]])
    
    return glist


def load_gold_queries(gold_path: str) -> List[List]:
    """
    Load gold queries from TSV format
    
    Format: SQL\\tDB_ID
    Empty lines separate sessions
    """
    with open(gold_path, 'r') as f:
        glist = []
        gseq_one = []
        
        for line in f.readlines():
            if len(line.strip()) == 0:
                if gseq_one:
                    glist.append(gseq_one)
                gseq_one = []
            else:
                lstrip = line.strip().split('\t')
                gseq_one.append(lstrip)
        
        if len(gseq_one) != 0:
            glist.append(gseq_one)
    
    return glist


def load_predictions(predict_path: str) -> List[List]:
    """Load predictions from TSV file"""
    if not predict_path:
        return []
    
    with open(predict_path, 'r') as f:
        plist = []
        pseq_one = []
        
        for line in f.readlines():
            if len(line.strip()) == 0:
                if pseq_one:
                    plist.append(pseq_one)
                pseq_one = []
            else:
                pseq_one.append(line.strip().split('\t'))
        
        if len(pseq_one) != 0:
            plist.append(pseq_one)
    
    return plist


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
    p_str = p_turn[0] if len(p_turn) > 0 else ""
    g_str = g_turn[0] if len(g_turn) > 0 else ""
    db_name = g_turn[1] if len(g_turn) > 1 else ""
    
    if not db_name:
        logger.error("No database name provided in gold data")
        return None
    
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return None
    
    # Normalize SQL
    if isinstance(p_str, str):
        p_str = normalize_sql_for_evaluation(p_str)
        p_str = p_str.strip('`').replace('`', '')
    
    if isinstance(g_str, str):
        g_str = normalize_sql_for_evaluation(g_str)
    
    # Parse gold SQL
    try:
        schema = load_schema(db_path)
        g_sql = parse_sql(g_str, schema)
    except Exception as e:
        logger.error(f"Error parsing gold SQL: {e}")
        logger.error(f"Gold SQL: {g_str}")
        return None
    
    hardness = eval_hardness(g_sql)
    
    # Parse predicted SQL
    try:
        p_sql = parse_sql(p_str, schema)
    except Exception as e:
        logger.warning(f"Parse failed: {e}")
        logger.debug(f"Failed SQL: {p_str}")
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
        try:
            exec_score = eval_exec_match(
                db=db_path, p_str=p_str, g_str=g_str,
                plug_value=plug_value, keep_distinct=keep_distinct,
                progress_bar_for_each_datapoint=progress_bar
            )
            result['exec_score'] = 1 if exec_score else 0
        except Exception as e:
            logger.warning(f"Execution evaluation failed: {e}")
            result['exec_score'] = 0
    
    # Exact match accuracy
    if etype in ["all", "match"]:
        try:
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            result['exact_score'] = 1 if exact_score else 0
            
            # Get partial scores
            if hasattr(evaluator, 'get_component_scores'):
                result['partial_scores'] = evaluator.get_component_scores()
        except Exception as e:
            logger.warning(f"Exact match evaluation failed: {e}")
            result['exact_score'] = 0
    
    return result


def initialize_scores() -> Dict:
    """Initialize score tracking structure"""
    return {
        'all': {'count': 0, 'exact': 0.0, 'exec': 0.0},
        'joint_all': {'count': 0, 'exact': 0, 'exec': 0}
    }


def finalize_scores(scores: Dict, etype: str):
    """Finalize and normalize scores"""
    count = scores['all']['count']
    if count > 0:
        scores['all']['exact'] /= count
        if etype in ['all', 'exec']:
            scores['all']['exec'] /= count


def create_evaluator(
    prompt_type: str = "enhanced",
    enable_debugging: bool = False,
    use_chromadb: bool = False,
    chromadb_config: Optional[Dict] = None,
    use_semantic: bool = False
) -> BaseEvaluator:
    """Create appropriate evaluator instance"""
    
    if use_chromadb:
        try:
            from src.evaluation.chromadb_evaluator import ChromaDBEvaluator
            return ChromaDBEvaluator(
                chromadb_config=chromadb_config,
                prompt_type=prompt_type,
                enable_debugging=enable_debugging
            )
        except ImportError:
            logger.warning("ChromaDB evaluator not available, using base evaluator")
    
    return BaseEvaluator()


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
    use_reasoning_bank: bool = False,
    reasoning_config: Optional[Dict] = None,
    limit: Optional[int] = None
) -> Dict:
    """
    Main evaluation function with Spider JSON support
    
    Args:
        gold: Path to gold queries file (JSON or TSV)
        predict: Path to predicted queries file (optional if using langchain)
        db_dir: Directory containing database files (data/raw/spider/database)
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
        semantic_config: Semantic layer configuration
        use_reasoning_bank: Use reasoning bank
        reasoning_config: Reasoning bank configuration
        limit: Limit number of examples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("="*80)
    logger.info("STARTING EVALUATION")
    logger.info("="*80)
    
    # === LOAD GOLD QUERIES ===
    if gold.endswith('.json'):
        logger.info(f"📊 Loading gold queries from Spider JSON: {gold}")
        glist = load_gold_from_spider_json(gold, limit=limit)
        logger.info(f"✓ Loaded {len(glist)} gold examples")
    else:
        logger.info(f"📊 Loading gold queries from TSV: {gold}")
        glist = load_gold_queries(gold)
        if limit:
            glist = glist[:limit]
            logger.info(f"✓ Limited to {limit} examples")
    
    # === CREATE EVALUATOR ===
    evaluator = create_evaluator(
        prompt_type=prompt_type,
        enable_debugging=enable_debugging,
        use_chromadb=use_chromadb,
        chromadb_config=chromadb_config,
        use_semantic=use_semantic
    )
    
    # === SEMANTIC PIPELINE ===
    semantic_pipeline = None
    if use_semantic and SEMANTIC_PIPELINE_AVAILABLE:
        try:
            semantic_pipeline = SemanticPipeline(semantic_config or {'enabled': True})
            logger.info("✓ Semantic pipeline enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic pipeline: {e}")
    
    # === GENERATE PREDICTIONS ===
    questions_list = []
    
    if use_langchain and questions_file:
        logger.info("🤖 Generating SQL from questions...")
        
        # Load questions
        examples = []
        if questions_file.endswith('.json'):
            with open(questions_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                if limit:
                    examples = examples[:limit]
        else:
            with open(questions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        examples.append({'question': parts[0], 'db_id': parts[1]})
                if limit:
                    examples = examples[:limit]
        
        logger.info(f"📝 Generating SQL for {len(examples)} questions...")
        
        # Initialize generator
        generator = None
        if not hasattr(evaluator, 'generate_sql_from_question'):
            try:
                from src.generation.sql_generator import SQLGenerator
                generator = SQLGenerator()
            except ImportError:
                logger.warning("SQLGenerator not available")
        
        # Generate predictions
        preds_list = []
        
        for i, example in enumerate(tqdm(examples, desc="Generating SQL")):
            question = example.get('question', '')
            questions_list.append(question)
            db_id = example.get('db_id', '')
            
            if not question or not db_id:
                logger.warning(f"⚠️  Example {i}: missing question or db_id")
                preds_list.append("SELECT 1")
                continue
            
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            
            # Check database exists
            if not os.path.exists(db_path):
                logger.error(f"❌ Database not found: {db_path}")
                logger.error(f"   Expected path: {db_dir}/{db_id}/{db_id}.sqlite")
                preds_list.append("SELECT 1")
                continue
            
            # Semantic enhancement
            enhanced_question = question
            if semantic_pipeline:
                try:
                    res = semantic_pipeline.enhance_question(question, db_id, None)
                    enhanced_question = res['enhanced_question']
                    if res['enhanced']:
                        logger.debug(f"✨ Enhanced: {len(res['suggestions'])} suggestions")
                except Exception as e:
                    logger.debug(f"Semantic enhancement skipped: {e}")
            
            # Generate SQL
            try:
                if hasattr(evaluator, 'generate_sql_from_question'):
                    sql = evaluator.generate_sql_from_question(enhanced_question, db_path)
                elif generator:
                    sql = generator.generate(enhanced_question, db_path)
                else:
                    logger.error("No SQL generator available")
                    sql = "SELECT 1"
                
                preds_list.append(sql)
                if enable_debugging:
                    logger.debug(f"[{i}] Q: {question[:50]}...")
                    logger.debug(f"[{i}] SQL: {sql[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Generation failed [{i}]: {e}")
                preds_list.append("SELECT 1")
        
        # Convert to evaluation format: [[[sql, db_id]], ...]
        plist = []
        for i, sql in enumerate(preds_list):
            db_id = examples[i].get('db_id', 'unknown')
            plist.append([[sql, db_id]])
        
    else:
        # Load predictions from file
        if not predict:
            raise ValueError("Either use --use_langchain with --questions or provide --pred file")
        logger.info(f"📊 Loading predictions from: {predict}")
        plist = load_predictions(predict)
        if limit:
            plist = plist[:limit]
    
    # === VALIDATE ALIGNMENT ===
    if len(plist) != len(glist):
        logger.error(f"❌ Length mismatch: predictions={len(plist)}, gold={len(glist)}")
        logger.warning("   Truncating to minimum length...")
        min_len = min(len(plist), len(glist))
        plist = plist[:min_len]
        glist = glist[:min_len]
        logger.info(f"✓ Proceeding with {min_len} examples")
    
    # === EVALUATE ===
    logger.info(f"\n🔍 Evaluating {len(plist)} predictions...")
    logger.info(f"   Evaluation type: {etype}")
    logger.info(f"   Database directory: {db_dir}\n")
    
    scores = initialize_scores()
    detailed_results = []
    
    for i in tqdm(range(len(plist)), desc="Evaluating", unit="query"):
        p_turn = plist[i][0]  # Extract [sql, db] from [[sql, db]]
        g_turn = glist[i][0]
        
        result = evaluate_turn(
            p_turn, g_turn, db_dir, etype, kmaps,
            evaluator, plug_value, keep_distinct,
            progress_bar_for_each_datapoint
        )
        
        if result:
            scores['all']['count'] += 1
            scores['all']['exact'] += result['exact_score']
            if etype in ['all', 'exec']:
                scores['all']['exec'] += result.get('exec_score', 0)
            
            # Collect detailed results
            detailed_results.append({
                'index': i,
                'question': questions_list[i] if i < len(questions_list) else 'N/A',
                'db_id': g_turn[1] if len(g_turn) > 1 else 'unknown',
                'gold_sql': result['entry']['goldSQL'],
                'predicted_sql': result['entry']['predictSQL'],
                'exact_match': bool(result['exact_score']),
                'execution_match': bool(result.get('exec_score', 0)),
                'hardness': result['hardness']
            })
    
    # === FINALIZE ===
    finalize_scores(scores, etype)
    
    # === PREPARE RESULTS ===
    results = {
        'exact_match_accuracy': scores['all']['exact'],
        'execution_accuracy': scores['all']['exec'] if etype in ['all', 'exec'] else 0.0,
        'total_evaluated': scores['all']['count'],
        'detailed_results': detailed_results,
        'scores': scores
    }
    
    # Add semantic statistics
    if use_semantic and SEMANTIC_PIPELINE_AVAILABLE and semantic_pipeline:
        try:
            results['semantic_statistics'] = semantic_pipeline.get_statistics()
        except:
            pass
    
    # === SUMMARY ===
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Total Evaluated: {results['total_evaluated']}")
    logger.info(f"✓ Exact Match Accuracy: {results['exact_match_accuracy']:.2%}")
    if etype in ['all', 'exec']:
        logger.info(f"✓ Execution Accuracy: {results['execution_accuracy']:.2%}")
    logger.info(f"{'='*80}\n")
    
    return results