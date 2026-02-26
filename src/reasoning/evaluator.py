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
import time
import re as _re
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path

from utils.sql_schema import Schema, load_schema, load_full_db_context, format_db_context_for_prompt
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


class RateLimiter:
    def __init__(self, requests_per_minute=30):
        self.rpm = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.time()


rate_limiter = RateLimiter(requests_per_minute=30)


# ---------------------------------------------------------------------------
# Utility: extract a valid SELECT statement from raw LLM output
# ---------------------------------------------------------------------------

def _extract_sql_from_output(text: str) -> str:
    """
    Robustly extract a clean SQL SELECT statement from arbitrary LLM output.

    Handles:
      - ```sql … ``` code fences
      - "SQL:" / "Final SQL:" label prefixes
      - DeepSeek / CoT reasoning prose surrounding the real query
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # 1. ```sql ... ``` block
    m = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return _finalize_sql(m.group(1))

    # 2. Generic ``` ... ``` that opens with SELECT
    m = re.search(r"```\s*(SELECT\b.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return _finalize_sql(m.group(1))

    # 3. Common label prefixes
    for prefix in (
        r"final\s+sql\s*query\s*:",
        r"final\s+sql\s*:",
        r"sql\s+query\s*:",
        r"sql\s*:",
        r"answer\s*:",
        r"query\s*:",
    ):
        m = re.search(prefix, text, re.IGNORECASE)
        if m:
            candidate = text[m.end():].strip()
            sql = _first_select(candidate)
            if sql:
                return _finalize_sql(sql)

    # 4. Last SELECT-starting line (prefer last — CoT draft then final)
    select_lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"SELECT\b", line.strip(), re.IGNORECASE)
    ]
    if select_lines:
        return _finalize_sql(select_lines[-1])

    # 5. First SELECT block anywhere
    sql = _first_select(text)
    if sql:
        return _finalize_sql(sql)

    return _finalize_sql(text)


def _first_select(text: str) -> str:
    """Pull the first complete SELECT statement from arbitrary text."""
    m = re.search(r"(SELECT\b.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _finalize_sql(sql: str) -> str:
    """Normalise extracted SQL: trim prose, remove backticks, collapse whitespace."""
    if not sql:
        return ""

    # Stop at first blank line
    sql = sql.split("\n\n")[0]
    # Remove trailing semicolon
    sql = sql.split(";")[0]
    # Drop DeepSeek prose footnotes: "But wait: ...", "However, note: ..."
    sql = re.sub(
        r"\s+\b(But|However|Note|Therefore|Also|Alternatively|Wait)\b[^\"']*$",
        "",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove backticks
    sql = sql.replace("`", "")
    # Collapse whitespace
    sql = " ".join(sql.split())
    return sql.strip()


def _is_valid_sql(sql: str) -> bool:
    """Quick check: does the string look like a real SQL statement?"""
    if not sql:
        return False
    return bool(re.match(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b", sql, re.IGNORECASE))


# ---------------------------------------------------------------------------


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
    """

    # === INITIALIZATION ===
    logger.info("=" * 80)
    logger.info("TEXT-TO-SQL EVALUATION WITH REASONINGBANK")
    logger.info("=" * 80)

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
            logger.error(f"ReasoningBank initialization error details: {e}", exc_info=True)

    # === LOAD DATA ===
    glist = []
    if gold.endswith('.json'):
        with open(gold, 'r') as f:
            data = json.load(f)
            for item in data:
                glist.append([item['query'], item['db_id']])
    else:
        glist = load_gold_queries(gold)

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
                schema_mapping[db_id] = load_schema(db_path)

    # === GENERATION OR LOAD PREDICTIONS ===
    plist = []
    trajectory_map: Dict[int, str] = {}  # index -> trajectory_id

    if use_langchain and questions_file:
        logger.info("🤖 Generating SQL with enhanced pipeline...")

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

        predictions = []
        failed_predictions = 0

        for i, item in enumerate(tqdm(questions_data, desc="Generating SQL")):
            rate_limiter.wait()

            question = item['question']
            db_id = item['db_id']
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            gold_sql = glist[i][0] if i < len(glist) else None

            try:
                db_context = load_full_db_context(db_id, db_dir)

                result = generate_sql_with_pipeline(
                    question=question,
                    db_id=db_id,
                    db_path=db_path,
                    schema=db_context['schema'],
                    gold_sql=gold_sql,
                    evaluator=evaluator,
                    semantic_pipeline=semantic_pipeline,
                    reasoning_pipeline=reasoning_pipeline
                )

                if result.get('trajectory_id'):
                    trajectory_map[i] = result['trajectory_id']

                sql = result.get('sql', '').strip()

                # ── FIX: post-process SQL if the LLM leaked prose ──────────
                if sql and not _is_valid_sql(sql):
                    logger.warning(f"[{i}] Non-SQL output detected, attempting extraction")
                    sql = _extract_sql_from_output(sql)
                    result['sql'] = sql

                if not sql:
                    logger.warning(f"Empty prediction for question {i}: {question}")
                    result['sql'] = "SELECT 1"
                    failed_predictions += 1

            except Exception as e:
                logger.error(f"SQL generation failed for question {i}: {e}")
                try:
                    raw = evaluator.generate_sql_from_question(question, db_path)
                    sql = _extract_sql_from_output(raw) if raw else "SELECT 1"
                    sql = sql if _is_valid_sql(sql) else "SELECT 1"
                except Exception:
                    sql = "SELECT 1"

                result = {'sql': sql, 'trajectory_id': None, 'error': str(e)}
                failed_predictions += 1

            predictions.append([result['sql']])

        if failed_predictions > 0:
            logger.warning(f"⚠️  {failed_predictions} predictions failed or were empty")

        plist = [predictions]

    elif predict:
        plist = load_predictions(predict)
        questions_data = []
    else:
        raise ValueError("Either use_langchain+questions_file or predict file required")

    # === EVALUATION ===
    evaluator_obj = BaseEvaluator()
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = [
        'select', 'select(no AGG)', 'where', 'where(no OP)',
        'group(no Having)', 'group', 'order', 'and/or', 'IUEN', 'keywords'
    ]

    scores = {}
    for level in levels:
        scores[level] = {'count': 0, 'exact': 0.0, 'exec': 0.0, 'partial': {}}
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0.0, 'rec': 0.0, 'f1': 0.0}

    detailed_results = []

    logger.info("📊 Evaluating predictions...")

    for i, (p_turn, g_turn) in enumerate(tqdm(
        zip(plist[0], glist),
        total=len(glist),
        desc="Evaluating"
    )):
        db_id = g_turn[1] if len(g_turn) > 1 else None

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
            hardness = res.get('hardness', 'all')
            for level in set(['all', hardness]):
                if level in scores:
                    scores[level]['count'] += 1
                    scores[level]['exact'] += res['exact_score']
                    scores[level]['exec'] += res.get('exec_score', 0)

                    if 'partial_scores' in res and res['partial_scores']:
                        for type_, p_score in res['partial_scores'].items():
                            if type_ in scores[level]['partial']:
                                scores[level]['partial'][type_]['acc'] += p_score['acc']
                                scores[level]['partial'][type_]['rec'] += p_score['rec']
                                scores[level]['partial'][type_]['f1'] += p_score['f1']

            # Update trajectory with evaluation results
            trajectory_id = trajectory_map.get(i)
            if reasoning_pipeline and trajectory_id:
                try:
                    reasoning_pipeline.update_trajectory(
                        trajectory_id=trajectory_id,
                        success=bool(res['exact_score']),
                        execution_success=res.get('exec_score', 0) > 0,
                        metadata={
                            'exact_match': bool(res['exact_score']),
                            'execution_match': res.get('exec_score', 0) > 0,
                            'hardness': res.get('hardness', 'unknown'),
                            'component_scores': res.get('component_scores'),
                            'partial_scores': res.get('partial_scores'),
                            'error': res.get('error'),
                            'parse_error': res.get('parse_error')
                        }
                    )
                    logger.debug(f"Updated trajectory {trajectory_id} with results")
                except Exception as e:
                    logger.warning(f"Failed to update trajectory {trajectory_id}: {e}")

            question = (
                questions_data[i]['question']
                if use_langchain and i < len(questions_data) else ''
            )
            detailed_results.append({
                'question': question,
                'gold_sql': res['entry']['goldSQL'],
                'predicted_sql': res['entry']['predictSQL'],
                'db_id': db_id,
                'exact_match': bool(res['exact_score']),
                'execution_match': res.get('exec_score', 0) > 0,
                'hardness': res.get('hardness', 'unknown'),
                'error': res.get('error'),
                'parse_error': res.get('parse_error'),
                'trajectory_id': trajectory_map.get(i)
            })

    # === FINALIZE SCORES ===
    for level in levels:
        count = scores[level]['count']
        if count > 0:
            scores[level]['exact'] /= count
            scores[level]['exec'] /= count
            for type_ in partial_types:
                scores[level]['partial'][type_]['acc'] /= count
                scores[level]['partial'][type_]['rec'] /= count
                scores[level]['partial'][type_]['f1'] /= count

    # === LEARNING PHASE ===
    distillation_result = {}
    consolidation_result = {}

    if reasoning_pipeline:
        logger.info("\n" + "=" * 80)
        logger.info("🧠 LEARNING PHASE: Distilling strategies from trajectories...")
        logger.info("=" * 80)

        try:
            distillation_result = reasoning_pipeline.distill_strategies()
            logger.info(f"✓ Strategy distillation complete")
            logger.info(f"  New strategies created: {distillation_result.get('new_strategies', 0)}")
            logger.info(f"  Strategies updated: {distillation_result.get('updated_strategies', 0)}")
            logger.info(f"  Trajectories processed: {len(trajectory_map)}")
        except Exception as e:
            logger.error(f"Failed to distill strategies: {e}")
            import traceback
            traceback.print_exc()
            distillation_result = {'error': str(e)}

        try:
            logger.info("\n💾 Consolidating memory...")
            consolidation_result = reasoning_pipeline.consolidate_memory()
            logger.info(f"✓ Memory consolidation complete")
            logger.info(f"  Patterns identified: {consolidation_result.get('patterns_identified', 0)}")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {e}")
            consolidation_result = {'error': str(e)}

    # === RESULTS ===
    results = {
        'scores': scores,
        'exact_match_accuracy': scores['all']['exact'],
        'execution_accuracy': scores['all']['exec'],
        'detailed_results': detailed_results
    }

    if semantic_pipeline:
        try:
            results['semantic_statistics'] = semantic_pipeline.get_statistics()
        except Exception as e:
            logger.warning(f"Failed to get semantic statistics: {e}")

    if reasoning_pipeline:
        try:
            reasoning_stats = reasoning_pipeline.get_statistics()
            reasoning_stats['distillation'] = distillation_result
            reasoning_stats['consolidation'] = consolidation_result
            reasoning_stats['total_trajectories_collected'] = len(trajectory_map)

            results['reasoning_statistics'] = reasoning_stats

            stats_file = './results/reasoning_stats.json'
            Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(reasoning_stats, f, indent=2)
            logger.info(f"Reasoning statistics saved to {stats_file}")
        except Exception as e:
            logger.warning(f"Failed to save reasoning statistics: {e}")

    try:
        output_file = './results/evaluation_results.json'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    print_scores(results['scores'], etype, include_turn_acc=False)

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
    Generate SQL with the complete pipeline.

    Flow:
      1. Semantic analysis (if enabled)
      2. ReasoningBank enhancement (if enabled)
      3. SQL extraction / cleanup
      4. Trajectory collection
    """

    # Step 1: Semantic analysis
    semantic_analysis = None
    if semantic_pipeline:
        try:
            semantic_analysis = semantic_pipeline.analyze(question)
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")

    # Step 2: ReasoningBank enhancement with retry logic
    if reasoning_pipeline:

        def sql_generator(q):
            max_retries = 3
            rate_limiter.wait()
            for attempt in range(max_retries):
                try:
                    raw = evaluator.generate_sql_from_question(q, db_path)

                    # Empty string means the generator gave up cleanly (prose
                    # output or LLM failure).  Don't retry — the model will
                    # produce the same prose again.  Return "" immediately so
                    # the ReasoningBank pipeline or outer caller can substitute
                    # SELECT 1.
                    if not raw or not raw.strip():
                        return ""

                    # If it looks like valid SQL, return it directly
                    if _is_valid_sql(raw):
                        return raw

                    # Try to recover SQL from prose output
                    sql = _extract_sql_from_output(raw)
                    if _is_valid_sql(sql):
                        return sql

                    # Extraction also failed — same model will keep producing
                    # prose, so give up immediately instead of burning retries.
                    logger.warning(
                        f"Could not extract SQL for: {q!r} — using SELECT 1"
                    )
                    return ""

                except Exception as e:
                    err = str(e)
                    if "429" in err or "Resource exhausted" in err:
                        if attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 30
                            logger.warning(
                                f"Rate limited, waiting {wait_time}s "
                                f"before retry {attempt + 1}/{max_retries}"
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error("Max retries reached for rate limiting")
                            return ""
                    else:
                        logger.error(f"Generation error: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            return ""
            return ""

        try:
            result = reasoning_pipeline.enhance_sql_generation(
                question=question,
                db_id=db_id,
                schema=schema,
                gold_sql=gold_sql,
                semantic_analysis=semantic_analysis,
                sql_generator=sql_generator
            )

            # Ensure the SQL in the result is clean
            raw_sql = result.get('sql', '')
            if raw_sql and not _is_valid_sql(raw_sql):
                extracted = _extract_sql_from_output(raw_sql)
                if extracted and _is_valid_sql(extracted):
                    result['sql'] = extracted
                    logger.debug(f"Post-processed SQL: {raw_sql[:80]!r} → {extracted!r}")
                else:
                    # Extraction failed — use safe fallback
                    logger.warning(
                        f"Could not extract SQL from output: {raw_sql[:80]!r}"
                    )
                    result['sql'] = "SELECT 1"

            if not result.get('sql'):
                result['sql'] = "SELECT 1"

            return result

        except Exception as e:
            logger.error(f"ReasoningBank generation failed: {e}")
            fallback_raw = sql_generator(question)
            fallback_sql = (
                _extract_sql_from_output(fallback_raw)
                if fallback_raw and not _is_valid_sql(fallback_raw)
                else fallback_raw
            ) or "SELECT 1"
            return {
                'sql': fallback_sql,
                'trajectory_id': None,
                'strategies_used': [],
                'generation_time': 0.0,
                'metadata': {'method': 'fallback', 'error': str(e)}
            }

    # Step 3: Standard generation (no ReasoningBank)
    try:
        raw = evaluator.generate_sql_from_question(question, db_path)
        sql = _extract_sql_from_output(raw) if raw and not _is_valid_sql(raw) else raw
        sql = sql or "SELECT 1"
    except Exception as e:
        logger.error(f"Standard generation failed: {e}")
        sql = "SELECT 1"

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
    schema = load_schema(db_path)

    # Check for placeholder queries
    if p_str.strip().upper() == "SELECT 1":
        return {
            'exact_score': 0,
            'exec_score': 0,
            'hardness': 'unknown',
            'entry': {'goldSQL': g_str, 'predictSQL': p_str},
            'error': 'placeholder_query',
            'partial_scores': {}
        }

    # Normalize queries
    p_str_normalized = normalize_sql_for_evaluation(p_str) if p_str else ""
    p_str_normalized = p_str_normalized.strip('`').replace('`', '')

    p_str_normalized = _re.sub(
        r'\bFROM\s+table\b', 'FROM wikisql_data', p_str_normalized, flags=_re.IGNORECASE
    )
    p_str_normalized = _re.sub(
        r'\bJOIN\s+table\b', 'JOIN wikisql_data', p_str_normalized, flags=_re.IGNORECASE
    )

    g_str_normalized = normalize_sql_for_evaluation(g_str) if g_str else ""

    # Parse SQL
    try:
        g_sql = get_sql(g_str_normalized, schema)
        p_sql = get_sql(p_str_normalized, schema)
    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown parse error"
        logger.warning(f"Parse error: {error_msg}")
        logger.debug(f"Gold SQL: {g_str}")
        logger.debug(f"Predict SQL: {p_str}")

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
            except Exception as exec_e:
                logger.debug(f"Execution evaluation also failed: {exec_e}")

        return {
            'exact_score': 0,
            'exec_score': exec_score,
            'hardness': 'unknown',
            'entry': {'goldSQL': g_str, 'predictSQL': p_str},
            'parse_error': error_msg,
            'partial_scores': {}
        }

    # Evaluate exact match
    exact_score = evaluator.eval_exact_match(p_sql, g_sql)
    partial_scores = evaluator.partial_scores

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
            logger.debug(f"Execution error: {e}")

    hardness = eval_hardness(g_sql)

    return {
        'exact_score': exact_score,
        'exec_score': exec_score,
        'hardness': hardness,
        'entry': {
            'goldSQL': g_str,
            'predictSQL': p_str
        },
        'component_scores': (
            evaluator.get_component_scores()
            if hasattr(evaluator, 'get_component_scores') else None
        ),
        'partial_scores': partial_scores
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


def load_schema_for_db(db_id: str, db_dir: str) -> Dict:
    """Load schema from database"""
    schema_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    schema = Schema(get_schema(schema_path))
    return schema.schema


def preprocess_question(question: str) -> str:
    """Normalize and clean question"""
    question = ' '.join(question.split())
    return question