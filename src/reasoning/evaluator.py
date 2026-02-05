"""
Enhanced Evaluator with Full ReasoningBank Integration

This is the updated evaluator.py that integrates:
1. SemanticPipeline
2. ReasoningBankPipeline
3. Complete evaluation flow with learning
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from utils.sql_schema import Schema, load_schema
from src.data.sql_parser import parse_sql
get_sql = parse_sql
from src.evaluation.base_evaluator import BaseEvaluator
from src.evaluation.hardness import eval_hardness
from src.evaluation.sql_rebuilder import (
    build_valid_col_units, rebuild_sql_col, rebuild_sql_val, clean_query
)
from src.evaluation.result_formatter import print_scores
from utils.logging_utils import get_logger
from utils.eval_utils import normalize_sql_for_evaluation

logger = get_logger(__name__)

# Import semantic pipeline
try:
    from src.semantic.semantic_pipeline import SemanticPipeline
    SEMANTIC_PIPELINE_AVAILABLE = True
except ImportError:
    SEMANTIC_PIPELINE_AVAILABLE = False
    logger.warning("Semantic pipeline not available")

# Import ReasoningBank pipeline
try:
    from src.reasoning.reasoning_pipeline import ReasoningBankPipeline
    REASONING_PIPELINE_AVAILABLE = True
except ImportError:
    REASONING_PIPELINE_AVAILABLE = False
    logger.warning("ReasoningBank pipeline not available")

# Import execution evaluator
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
    use_reasoning_bank: bool = False,
    reasoning_config: Optional[Dict] = None,
    limit: Optional[int] = None
) -> Dict:
    """
    Main evaluation function with full pipeline integration
    
    New parameters:
        use_reasoning_bank: Enable ReasoningBank pipeline
        reasoning_config: Configuration for ReasoningBank
    """
    
    # === INITIALIZATION ===
    logger.info("="*80)
    logger.info("TEXT-TO-SQL EVALUATION WITH REASONINGBANK")
    logger.info("="*80)
    
    # Create base evaluator
    evaluator = create_evaluator(
        prompt_type=prompt_type,
        enable_debugging=enable_debugging,
        use_chromadb=use_chromadb,
        chromadb_config=chromadb_config,
        use_semantic=use_semantic
    )
    
    # Initialize Semantic Pipeline
    semantic_pipeline = None
    if use_semantic and SEMANTIC_PIPELINE_AVAILABLE:
        try:
            semantic_pipeline = SemanticPipeline(semantic_config or {'enabled': True})
            logger.info("✓ Semantic pipeline enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic pipeline: {e}")
    
    # Initialize ReasoningBank Pipeline
    reasoning_pipeline = None
    if use_reasoning_bank and REASONING_PIPELINE_AVAILABLE:
        try:
            reasoning_pipeline = ReasoningBankPipeline(
                db_path="./memory/reasoning_bank.db",
                chromadb_path="./memory/chromadb",
                config=reasoning_config
            )
            logger.info("✓ ReasoningBank pipeline enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize ReasoningBank pipeline: {e}")
    
    # === LOAD DATA ===
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
    
    # Load schema mapping
    schema_mapping = {}
    for g_turn in glist:
        db_id = g_turn[1] if len(g_turn) > 1 else None
        if db_id and db_id not in schema_mapping:
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                schema_mapping[db_id] = load_schema(db_id, db_dir)
    
    # === GENERATION OR LOAD PREDICTIONS ===
    plist = []
    
    if use_langchain and questions_file:
        # Generate predictions with full pipeline
        logger.info("🤖 Generating SQL with enhanced pipeline...")
        
        # Load questions
        with open(questions_file, 'r') as f:
            if questions_file.endswith('.json'):
                questions_data = json.load(f)
            else:
                questions_data = [
                    {'question': line.strip().split('\t')[0], 
                     'db_id': line.strip().split('\t')[1]}
                    for line in f if line.strip()
                ]
        
        if limit:
            questions_data = questions_data[:limit]
        
        # Generate with pipeline
        predictions = []
        for i, item in enumerate(tqdm(questions_data, desc="Generating SQL")):
            question = item['question']
            db_id = item['db_id']
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            schema = schema_mapping.get(db_id)
            gold_sql = glist[i][0] if i < len(glist) else None
            
            # Generate with full pipeline
            result = generate_sql_with_pipeline(
                question=question,
                db_id=db_id,
                db_path=db_path,
                schema=schema,
                gold_sql=gold_sql,
                evaluator=evaluator,
                semantic_pipeline=semantic_pipeline,
                reasoning_pipeline=reasoning_pipeline
            )
            
            predictions.append([result['sql']])
        
        plist = [predictions]
    
    elif predict:
        # Load existing predictions
        plist = load_predictions(predict)
    else:
        raise ValueError("Either use_langchain+questions_file or predict file required")
    
    # === EVALUATION ===
    evaluator_obj = BaseEvaluator()
    scores = {'all': {'count': 0, 'exact': 0.0, 'exec': 0.0}}
    detailed_results = []
    
    logger.info("📊 Evaluating predictions...")
    
    for i, (p_turn, g_turn) in enumerate(tqdm(
        zip(plist[0], glist),
        total=len(glist),
        desc="Evaluating"
    )):
        db_id = g_turn[1] if len(g_turn) > 1 else None
        
        # Evaluate turn
        res = evaluate_turn(
            p_turn=p_turn,
            g_turn=g_turn,
            db_dir=db_dir,
            etype=etype,
            kmaps=kmaps,
            evaluator=evaluator_obj,
            plug_value=plug_value,
            keep_distinct=keep_distinct,
            progress_bar=progress_bar_for_each_datapoint
        )
        
        if res:
            # Update scores
            scores['all']['count'] += 1
            scores['all']['exact'] += res['exact_score']
            scores['all']['exec'] += res.get('exec_score', 0)
            
            # Process with ReasoningBank
            if reasoning_pipeline and 'trajectory_id' in res:
                reasoning_pipeline.process_evaluation_result(
                    trajectory_id=res['trajectory_id'],
                    exact_match=res['exact_score'],
                    execution_match=res.get('exec_score', 0) > 0,
                    component_scores=res.get('component_scores')
                )
            
            # Collect detailed results
            detailed_results.append({
                'question': questions_data[i]['question'] if use_langchain else '',
                'gold_sql': g_turn[0],
                'predicted_sql': p_turn[0] if p_turn else '',
                'db_id': db_id,
                'exact_match': bool(res['exact_score']),
                'execution_match': res.get('exec_score', 0) > 0,
                'hardness': res.get('hardness', 'unknown')
            })
    
    # === FINALIZE ===
    finalize_scores(scores, etype)
    
    # Consolidate learning if using ReasoningBank
    if reasoning_pipeline:
        logger.info("🧠 Final memory consolidation...")
        reasoning_pipeline.consolidate_memory()
    
    # === RESULTS ===
    results = {
        'scores': scores,
        'exact_match_accuracy': scores['all']['exact'],
        'execution_accuracy': scores['all']['exec'],
        'detailed_results': detailed_results
    }
    
    # Add semantic statistics
    if semantic_pipeline:
        try:
            results['semantic_statistics'] = semantic_pipeline.get_statistics()
        except:
            pass
    
    # Add ReasoningBank statistics
    if reasoning_pipeline:
        try:
            results['reasoning_statistics'] = reasoning_pipeline.get_statistics()
            reasoning_pipeline.save_statistics('./results/reasoning_stats.json')
        except:
            pass
    
    # Save detailed results
    try:
        output_file = './results/evaluation_results.json'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Print summary
    print_scores(results, etype)
    
    return results


def generate_sql_with_pipeline(
    question: str,
    db_id: str,
    db_path: str,
    schema: Dict,
    gold_sql: Optional[str],
    evaluator,
    semantic_pipeline: Optional['SemanticPipeline'],
    reasoning_pipeline: Optional['ReasoningBankPipeline']
) -> Dict:
    """
    Generate SQL with complete pipeline
    
    Flow:
    1. Semantic analysis (if enabled)
    2. ReasoningBank enhancement (if enabled)
    3. SQL generation
    4. Trajectory collection
    
    Returns:
        Dictionary with SQL and metadata
    """
    
    # Step 1: Semantic analysis
    semantic_analysis = None
    if semantic_pipeline:
        try:
            semantic_analysis = semantic_pipeline.analyze(question)
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
    
    # Step 2: ReasoningBank enhancement
    if reasoning_pipeline:
        # Define SQL generator function
        def sql_generator(q):
            return evaluator.generate_sql_from_question(q, db_path)
        
        # Generate with ReasoningBank
        result = reasoning_pipeline.enhance_sql_generation(
            question=question,
            db_id=db_id,
            schema=schema,
            gold_sql=gold_sql,
            semantic_analysis=semantic_analysis,
            sql_generator=sql_generator
        )
        
        return result
    
    # Step 3: Standard generation (fallback)
    else:
        sql = evaluator.generate_sql_from_question(question, db_path)
        return {
            'sql': sql,
            'trajectory_id': None,
            'strategies_used': [],
            'generation_time': 0.0,
            'metadata': {'method': 'standard'}
        }


def create_evaluator(
    prompt_type: str,
    enable_debugging: bool,
    use_chromadb: bool,
    chromadb_config: Optional[Dict],
    use_semantic: bool
):
    """Create appropriate evaluator based on configuration"""
    
    # Try semantic evaluator first
    if use_semantic:
        try:
            from semantic_layer import SemanticEvaluator
            return SemanticEvaluator(
                prompt_type=prompt_type,
                enable_debugging=enable_debugging,
                use_chromadb=use_chromadb,
                chromadb_config=chromadb_config
            )
        except ImportError:
            logger.warning("SemanticEvaluator not available, using base evaluator")
    
    # Try ChromaDB evaluator
    if use_chromadb:
        try:
            from src.evaluation.chromadb_evaluator import ChromaDBEvaluator
            return ChromaDBEvaluator(
                prompt_type=prompt_type,
                enable_debugging=enable_debugging,
                chromadb_config=chromadb_config
            )
        except ImportError:
            logger.warning("ChromaDBEvaluator not available, using base evaluator")
    
    # Base evaluator
    return BaseEvaluator()


def evaluate_turn(
    p_turn, g_turn, db_dir: str, etype: str, kmaps: Dict,
    evaluator, plug_value: bool, keep_distinct: bool,
    progress_bar: bool
) -> Optional[Dict]:
    """Evaluate a single turn"""
    
    p_str = p_turn[0] if len(p_turn) > 0 else ""
    g_str = g_turn[0] if len(g_turn) > 0 else ""
    db_name = g_turn[1] if len(g_turn) > 1 else None
    
    if not db_name:
        return None
    
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    schema = load_schema(db_name, db_dir)
    
    # Normalize queries
    p_str_normalized = normalize_sql_for_evaluation(p_str)
    g_str_normalized = normalize_sql_for_evaluation(g_str)
    
    # Parse SQL
    try:
        g_sql = get_sql(schema, g_str_normalized)
        p_sql = get_sql(schema, p_str_normalized)
    except Exception as e:
        logger.warning(f"Parse error: {e}")
        return {
            'exact_score': 0,
            'exec_score': 0,
            'entry': {'goldSQL': g_str, 'predictSQL': p_str}
        }
    
    # Evaluate exact match
    exact_score = evaluator.eval_exact_match(p_sql, g_sql)
    
    # Evaluate execution
    exec_score = 0
    if EXEC_EVAL_AVAILABLE and etype in ['all', 'exec']:
        try:
            exec_score = eval_exec_match(
                db=db_path,
                p_str=p_str_normalized,
                g_str=g_str_normalized,
                plug_value=plug_value,
                keep_distinct=keep_distinct,
                progress_bar_for_each_datapoint=progress_bar
            )
        except Exception as e:
            logger.warning(f"Execution error: {e}")
    
    # Get hardness
    hardness = eval_hardness(g_sql)
    
    return {
        'exact_score': exact_score,
        'exec_score': exec_score,
        'hardness': hardness,
        'entry': {
            'goldSQL': g_str,
            'predictSQL': p_str
        },
        'component_scores': evaluator.get_component_scores() if hasattr(evaluator, 'get_component_scores') else None
    }


def load_predictions(predict_path: str) -> List[List]:
    """Load predictions from file"""
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


def finalize_scores(scores: Dict, etype: str):
    """Finalize and normalize scores"""
    count = scores['all']['count']
    if count > 0:
        scores['all']['exact'] /= count
        if etype in ['all', 'exec']:
            scores['all']['exec'] /= count