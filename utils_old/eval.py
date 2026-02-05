################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import os, sys
import argparse
import sqlite3
import re
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_sql import get_schema, Schema, get_sql
from exec_eval import eval_exec_match
from dotenv import load_dotenv

# Import template manager from separate file
try:
    from template_manager import TemplateManager, PromptType, SpecializedPrompts
    TEMPLATE_MANAGER_AVAILABLE = True
    print("✅ Template manager available")
except ImportError:
    TEMPLATE_MANAGER_AVAILABLE = False
    print("⚠️  Template manager not available. Using basic prompts.")

# ChromaDB integration
try:
    #from spider_chromadb_integration import SpiderChromaDBIntegration
    from interactive_spider_query import InteractiveSpiderQuery
    CHROMADB_AVAILABLE = True
    print("✅ ChromaDB integration available")
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB integration not available")
    
# LangChain integration (optional)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import google.generativeai as genai
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain available for text-to-SQL generation")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: pip install langchain langchain-openai")

try:
    from semantic_layer import SemanticEvaluator
    SEMANTIC_AVAILABLE = True
    print("✅ Semantic layer available")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️  Semantic layer not available")

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}

def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res

def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0

def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct

def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2

def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql

def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2

def rebuild_condition_col(valid_col_units, condition, kmap):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition

def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_

def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]

def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    return direction, new_val_units

def rebuild_select_col(valid_col_units, sel, kmap):
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

def rebuild_sql_col(valid_col_units, sql, kmap):
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

def rebuild_cond_unit_val(cond_unit):
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

def build_valid_col_units(table_units, schema):
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units= []
    for value in schema.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units

def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res

def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql

def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg

def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg

def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt

def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt

def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt

def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0

def eval_nested(pred, label, evaluator):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += evaluator.eval_exact_match(pred, label)
    return label_total, pred_total, cnt

def eval_IUEN(pred, label, evaluator):
    """Evaluate INTERSECT, UNION, EXCEPT operations"""
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'], evaluator)
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'], evaluator)
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'], evaluator)
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt

def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt

def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')

def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])

def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested

def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count

def print_formated_s(row_name, l, element_format):
    template = "{:20} " + ' '.join([element_format] * len(l))
    print(template.format(row_name, *l))

def print_scores(scores, etype, include_turn_acc=True):
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    if include_turn_acc:
        levels.append('joint_all')
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']

    print_formated_s("", levels, '{:20}')
    counts = [scores[level]['count'] for level in levels]
    print_formated_s("count", counts, '{:<20d}')

    if etype in ["all", "exec"]:
        print ('=====================   EXECUTION ACCURACY     =====================')
        exec_scores = [scores[level]['exec'] for level in levels]
        print_formated_s("execution", exec_scores, '{:<20.3f}')

    if etype in ["all", "match"]:
        print ('\n====================== EXACT MATCHING ACCURACY =====================')
        exact_scores = [scores[level]['exact'] for level in levels]
        print_formated_s("exact match", exact_scores, '{:<20.3f}')
        print ('\n---------------------PARTIAL MATCHING ACCURACY----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            print_formated_s(type_, this_scores, '{:<20.3f}')

        print ('---------------------- PARTIAL MATCHING RECALL ----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            print_formated_s(type_, this_scores, '{:<20.3f}')

        print ('---------------------- PARTIAL MATCHING F1 --------------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
            print_formated_s(type_, this_scores, '{:<20.3f}')

    if include_turn_acc:
        print()
        print()
        print_formated_s("", turns, '{:20}')
        counts = [scores[turn]['count'] for turn in turns]
        print_formated_s("count", counts, "{:<20d}")

        if etype in ["all", "exec"]:
            print ('=====================   TURN EXECUTION ACCURACY     =====================')
            exec_scores = [scores[turn]['exec'] for turn in turns]
            print_formated_s("execution", exec_scores, '{:<20.3f}')

        if etype in ["all", "match"]:
            print ('\n====================== TURN EXACT MATCHING ACCURACY =====================')
            exact_scores = [scores[turn]['exact'] for turn in turns]
            print_formated_s("exact match", exact_scores, '{:<20.3f}')

def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)

def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count

def clean_querry(sql_dict):
    """
    Recursively clean backticks from SQL query dictionary
    """
    if isinstance(sql_dict, dict):
        cleaned = {}
        for key, value in sql_dict.items():
            cleaned[key] = clean_querry(value)
        return cleaned
    elif isinstance(sql_dict, list):
        return [clean_querry(item) for item in sql_dict]
    elif isinstance(sql_dict, str):
        # Remove backticks from string values
        return sql_dict.strip('`').replace('`', '')
    else:
        return sql_dict
    
def evaluate(gold, predict, db_dir, etype, kmaps, plug_value, keep_distinct, progress_bar_for_each_datapoint, 
             use_langchain=False, questions_file=None, prompt_type="enhanced", enable_debugging=False,
             use_chromadb=False, chromadb_config=None, use_semantic = False):
    # LangChain mode: generate predictions from questionsFaFa
    if use_langchain and questions_file:
        print("🤖 LangChain Mode: Generating SQL from natural language questions")
        
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available. Please install: pip install langchain langchain-google-genai")
            return {}
        
        # Load questions and generate predictions
        evaluator = create_evaluator(
            prompt_type=prompt_type, 
            enable_debugging=enable_debugging,
            use_chromadb=use_chromadb,
            chromadb_config=chromadb_config, 
            use_semantic=use_semantic
        )
        
        with open(questions_file, 'r') as f:
            question_lines = f.readlines()
        
        generated_predictions = []
        for line_num, line in enumerate(question_lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parse question and database name with improved logic
            parts = None
            if '\t' in line:
                parts = line.split('\t')
            elif '  ' in line:  # Two or more spaces
                parts = re.split(r'\s{2,}', line)  # Split on 2+ consecutive spaces
            elif '\\t' in line:
                parts = line.split('\\t')
            elif ' ' in line:
                parts = line.rsplit(' ', 1)  # rsplit from right, maxsplit=1
        
            if not parts or len(parts) < 2:
                print(f"⚠️  Skipping line {line_num}: Could not parse '{line}'")
                continue
                
            question = parts[0].strip()
            db_name = parts[1].strip()
            
            print(f"📝 Line {line_num}: Question='{question}' DB='{db_name}'")
            
            # Construct correct database path
            db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
            
            # Generate SQL using LangChain
            generated_sql = evaluator.generate_sql_from_question(question, db_path)
            generated_predictions.append([generated_sql])
            
            print(f"Q: {question[:50]}...")
            print(f"Generated SQL: {generated_sql}")
        
        # Create temporary prediction list for evaluation
        plist = [generated_predictions]
        
        print(f"✅ Generated {len(generated_predictions)} SQL predictions using LangChain")

        if use_semantic and SEMANTIC_AVAILABLE and hasattr(evaluator, 'get_semantic_statistics'):
            try:
                print("\n" + "=" * 60)
                print("📊 SEMANTIC LAYER ANALYSIS SUMMARY")
                print("=" * 60)
                
                stats = evaluator.get_semantic_statistics()
                
                print(f"\n📈 Query Analysis:")
                print(f"   Queries Analyzed: {stats.get('queries_analyzed', 0)}")
                print(f"   Queries Enhanced: {stats.get('queries_enhanced', 0)}")
                print(f"   Suggestions Made: {stats.get('suggestions_made', 0)}")
                
                if stats.get('complexity_scores'):
                    avg_complexity = sum(stats['complexity_scores']) / len(stats['complexity_scores'])
                    print(f"   Average Complexity: {avg_complexity:.2f}")
                
                print(f"\n🔧 Enhancements Applied:")
                enhancement_types = stats.get('enhancement_types', {})
                for enhancement, count in sorted(enhancement_types.items()):
                    if count > 0:
                        name = enhancement.replace('_', ' ').title()
                        print(f"   {name}: {count}")
                
                print(f"\n🎯 Entity Recognition:")
                entity_detections = stats.get('entity_detections', {})
                total_entities = sum(entity_detections.values())
                if total_entities > 0:
                    for entity, count in sorted(entity_detections.items()):
                        if count > 0:
                            pct = (count / total_entities) * 100
                            name = entity.replace('_', ' ').title()
                            print(f"   {name}: {count} ({pct:.1f}%)")
                
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"Warning: Could not retrieve semantic statistics: {e}")

        # Print ChromaDB statistics if available
        if use_chromadb and hasattr(evaluator, 'get_retrieval_statistics'):
            retrieval_stats = evaluator.get_retrieval_statistics()
            print("\n📊 ChromaDB Retrieval Statistics:")
            print(f"  Queries with retrieval: {retrieval_stats.get('queries_with_retrieval', 0)}")
            print(f"  Successful retrievals: {retrieval_stats.get('successful_retrievals', 0)}")
            print(f"  Retrieval helped: {retrieval_stats.get('retrieval_helped', 0)}")
            print(f"  Average similarity: {retrieval_stats.get('average_similarity', 0.0):.3f}")
    else:
        with open(predict) as f:
            plist = []
            pseq_one = []
            for l in f.readlines():
                if len(l.strip()) == 0:
                    plist.append(pseq_one)
                    pseq_one = []
                else:
                    pseq_one.append(l.strip().split('\t'))

            if len(pseq_one) != 0:
                plist.append(pseq_one)

    with open(gold) as f:
        glist = []
        gseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                glist.append(gseq_one)
                gseq_one = []
            else:
                lstrip = l.strip().split('\t')
                gseq_one.append(lstrip)

        if len(gseq_one) != 0:
            glist.append(gseq_one)

    # spider formatting indicates that there is only one "single turn"
    # do not report "turn accuracy" for SPIDER
    include_turn_acc = len(glist) > 1

    assert len(plist) == len(glist), "number of sessions must equal"

    if use_chromadb and CHROMADB_AVAILABLE and TEMPLATE_MANAGER_AVAILABLE:
        evaluator = ChromaDBEvaluator(
            prompt_type=PromptType.ENHANCED,
            enable_debugging=False,
            use_chromadb=use_chromadb,
            chromadb_config=chromadb_config
        )
    else:
        evaluator = BaseEvaluator()

    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all', 'joint_all']

    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for turn in turns:
        scores[turn] = {'count': 0, 'exact': 0.}
        scores[turn]['exec'] = 0

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0.,'acc_count':0,'rec_count':0}

    langchain_stats = {'generated': 0, 'successful_parse': 0, 'exact_matches': 0}

    # for i, (p, g) in tqdm(enumerate(zip(plist, glist)), desc='Compare Pred SQL with Gold SQL', total=len(plist)):
    for i, (p, g) in enumerate(zip(plist, glist)):
        scores['joint_all']['count'] += 1
        turn_scores = {"exec": [], "exact": []}

        for idx, pg in tqdm(enumerate(zip(p, g))):
            p, g = pg
            p_str = p[0] if len(p) > 0 else ""
            g_str, db = g
            db_name = db
            db_path = os.path.join(db_dir, db, db + ".sqlite")  # Use consistent db_path

            if use_langchain:
                langchain_stats['generated'] += 1
                print(f"🔍 Evaluating: {p_str[:50]}...")
            try:
                schema = load_schema(db_path)
                g_sql = get_sql(schema, g_str)
            except Exception as e:
                print(f"❌ Error parsing gold SQL: {e}")
                print(f"   Database path: {db_path}")
                print(f"   Gold SQL: {g_str}")
                continue

            hardness = evaluator.eval_hardness(g_sql)
            turn_id = "turn " + ("> 4" if idx > 3 else str(idx + 1))
            scores[turn_id]['count'] += 1
            scores[hardness]['count'] += 1
            scores['all']['count'] += 1

            if isinstance(p_str, str):
                p_str = p_str.strip('`').replace('`', '')
            try:
                p_sql = get_sql(schema, p_str)
                if use_langchain:
                    langchain_stats['successful_parse'] += 1
            except Exception as e:
                if use_langchain:
                    print(f"⚠️  Parse failed: {e}")
                # If p_sql is not valid, use an empty sql structure

                 # SAVE FAILED QUERY TO FILE
                with open("../questions/pred_queries_comprehensive_failed.txt", "a", encoding="utf-8") as f:
                    f.write(f"PARSE FAILED: {e}\n")
                    f.write(f"Predicted: {p_str}\n")
                    f.write(f"Gold: {g_str}\n")
                    f.write(f"Database: {db_name}\n")
                    f.write("-" * 80 + "\n")

                p_sql = {
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

            # Method execution accuracy (EX)
            if etype in ["all", "exec"]:
                exec_score = eval_exec_match(db=db_path, p_str=p_str, g_str=g_str, plug_value=plug_value,
                                             keep_distinct=keep_distinct, progress_bar_for_each_datapoint=progress_bar_for_each_datapoint)
                print("exec_score", exec_score)
                if exec_score:
                    scores[hardness]['exec'] += 1
                    scores[turn_id]['exec'] += 1
                    scores['all']['exec'] += 1
                    turn_scores['exec'].append(1)
                else:
                    turn_scores['exec'].append(0)
                    print("EX fail")
                    print("id: {}-{}".format(i, idx))
                    print("{} pred: {}".format(hardness, p_str))
                    print("{} gold: {}".format(hardness, g_str))
                    print("")
                    
            # Method exact match accuracy (EM),
            if etype in ["all", "match"]:
                # rebuild sql for value evaluation
                kmap = kmaps[db_name]
                g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
                g_sql = rebuild_sql_val(g_sql)
                g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
                p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
                p_sql = rebuild_sql_val(p_sql)
                p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
                p_sql = clean_querry(p_sql) 
                exact_score = evaluator.eval_exact_match(p_sql, g_sql)

                # Save p_str to a text file
                with open("../questions/pred_queries.txt", "a", encoding="utf-8") as f:
                    f.write(p_str + "\n")
                partial_scores = evaluator.partial_scores
                if exact_score == 0:
                    turn_scores['exact'].append(0)
                    if use_langchain and (etype == "match" or (etype == "all" and turn_scores['exec'][-1] == 1)):
                        print("❌ EM fail")
                        print(f"   Pred: {p_str}")
                        print(f"   Gold: {g_str}")
                        print()
                      
                else:
                    turn_scores['exact'].append(1)
                    if use_langchain:
                        langchain_stats['exact_matches'] += 1
                        print("✅ Exact match!")
              
                scores[turn_id]['exact'] += exact_score
                scores[hardness]['exact'] += exact_score
                scores['all']['exact'] += exact_score
                for type_ in partial_types:
                    if partial_scores[type_]['pred_total'] > 0:
                        scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores[hardness]['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores[hardness]['partial'][type_]['rec_count'] += 1
                    scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                    if partial_scores[type_]['pred_total'] > 0:
                        scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores['all']['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores['all']['partial'][type_]['rec_count'] += 1
                    scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

                entries.append({
                    'predictSQL': p_str,
                    'goldSQL': g_str,
                    'hardness': hardness,
                    'exact': exact_score,
                    'partial': partial_scores
                })

        if all(v == 1 for v in turn_scores["exec"]):
            scores['joint_all']['exec'] += 1

        if all(v == 1 for v in turn_scores["exact"]):
            scores['joint_all']['exact'] += 1

    # Calculate final scores
    for turn in turns:
        if scores[turn]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[turn]['exec'] /= scores[turn]['count']

        if etype in ["all", "match"]:
            scores[turn]['exact'] /= scores[turn]['count']

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                                                             scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                                                             scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                        scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    # Print results
    if use_langchain:
        print("\n🤖 LANGCHAIN EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total questions processed: {langchain_stats['generated']}")
        print(f"Successfully parsed SQL: {langchain_stats['successful_parse']}")
        print(f"Exact matches: {langchain_stats['exact_matches']}")
        if langchain_stats['generated'] > 0:
            parse_rate = langchain_stats['successful_parse'] / langchain_stats['generated'] * 100
            match_rate = langchain_stats['exact_matches'] / langchain_stats['generated'] * 100
            print(f"Parse success rate: {parse_rate:.1f}%")
            print(f"Exact match rate: {match_rate:.1f}%")
        print()

    print_scores(scores, etype, include_turn_acc=include_turn_acc)

    if etype == 'match': return {'exact': scores['all']['exact']}
    elif etype == 'exec': return {'exec': scores['all']['exec']}
    else: return {'exact': scores['all']['exact'], 'exec': scores['all']['exec']}

def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

     # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        # Handle different formats in Spider dataset
        if isinstance(col_orig, list) and len(col_orig) == 2:
            table_idx, col_name = col_orig
            # Check if table_idx is a valid integer and >= 0
            if isinstance(table_idx, int) and table_idx >= 0:
                t = tables_orig[table_idx]
                c = col_name
                cols.append("__" + t.lower() + "." + c.lower() + "__")
            else:
                # Handle special cases like [-1, "*"] or ["*"]
                cols.append("__all__")
        elif isinstance(col_orig, list) and len(col_orig) == 1:
            # Handle cases like ["*"]
            cols.append("__all__")
        else:
            # Fallback for unexpected formats
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
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

def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables

def create_evaluator(prompt_type="enhanced", enable_debugging=False, use_chromadb=False, chromadb_config=None, use_semantic = False):
    """Create an evaluator with specified prompt type and ChromaDB options"""
    if TEMPLATE_MANAGER_AVAILABLE:
        # Convert string to PromptType enum
        if isinstance(prompt_type, str):
            prompt_type_map = {
                'basic': PromptType.BASIC,
                'few_shot': PromptType.FEW_SHOT,
                'chain_of_thought': PromptType.CHAIN_OF_THOUGHT,
                'rule_based': PromptType.RULE_BASED,
                'enhanced': PromptType.ENHANCED,
                'step_by_step': PromptType.STEP_BY_STEP
            }
            prompt_type = prompt_type_map.get(prompt_type, PromptType.ENHANCED)
        
        if use_semantic and SEMANTIC_AVAILABLE:
            print("🎯 Using Semantic Enhanced Evaluator")
            return SemanticEvaluator(
                prompt_type=prompt_type,
                enable_debugging=enable_debugging,
                use_chromadb=use_chromadb,
                chromadb_config=chromadb_config
            )
        # Use ChromaDBEvaluator if ChromaDB is requested
        elif use_chromadb and CHROMADB_AVAILABLE:
            return ChromaDBEvaluator(
                prompt_type=prompt_type, 
                enable_debugging=enable_debugging,
                use_chromadb=use_chromadb,
                chromadb_config=chromadb_config
            )
        else:
            return BaseEvaluator()
    else:
        print("⚠️  Template manager not available, falling back to basic evaluator")
        return BaseEvaluator()
    
def normalize_sql_for_evaluation(sql):
        """Normalize SQL query for fair comparison."""
        if not sql:
            return sql
        
        # Remove newlines and normalize whitespace
        normalized = ' '.join(sql.strip().split())
        
        # Remove trailing semicolon
        if normalized.endswith(';'):
            normalized = normalized[:-1]
        
        return normalized

class BaseEvaluator:
    """Base evaluator class with original functionality"""
    def __init__(self):
        self.partial_scores = None
        self.langchain_generator = None
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()

    def normalize_sql_structure(self, sql_dict):
        """
        Normalize SQL structure dictionary for comparison.
        This handles alias normalization and structural standardization.
        """
        if not sql_dict:
            return sql_dict
        
        # For now, return as-is since the parsing already handles most normalization
        # You can add more sophisticated normalization here if needed
        return sql_dict
    
    def _setup_langchain(self):
        """Setup basic LangChain for text-to-SQL generation"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, "..", ".env")
        load_dotenv(env_path)
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your-api-key-here":
            print("⚠️  Google API key not found. LangChain will use pattern matching.")
            return
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
            
            # Spider-compatible examples - NO ALIASES
            examples = [
                {"question": "How many students are there?", "sql": "SELECT COUNT(*) FROM student"},
                {"question": "List all movies", "sql": "SELECT * FROM movie"},
                {"question": "What are the teacher names?", "sql": "SELECT name FROM teacher"},
                {"question": "Show breed and size codes from dogs", "sql": "SELECT DISTINCT breed_code, size_code FROM dogs"},
                {"question": "Show singers born in 1948 or 1949", "sql": "SELECT name FROM singer WHERE birth_year = 1948 OR birth_year = 1949"},
            ]
            
            example_prompt = ChatPromptTemplate.from_messages([
                ("human", "Question: {question}"),
                ("ai", "SQL: {sql}")
            ])
            
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=examples
            )
            
            system_prompt = """You are a SQL expert for the Spider benchmark. Generate the SIMPLEST possible query.

    Database Schema: {schema}

    CRITICAL RULES:
    1. NEVER use table aliases of ANY kind (no AS T1, no single letters like 'b' or 's', NOTHING)
    2. Start with the SIMPLEST solution - always prefer single-table queries
    3. Only use JOINs if absolutely necessary to get the requested data
    4. Check if ONE table already contains all needed columns before joining
    5. Column references in single-table queries: use column_name only
    6. Column references in multi-table queries: use Table.column_name format
    7. For codes/IDs: Select the code column, not the description column
    8. For OR conditions: Use "col = val1 OR col = val2", NOT "IN (val1, val2)"
    9. NEVER use LEFT JOIN, RIGHT JOIN, or OUTER JOIN

    DECISION TREE:
    Step 1: Does ONE table have all the columns needed? → Use that table alone, no JOINs
    Step 2: Need data from multiple tables? → Use JOIN with Table.column format (NO aliases)

    CORRECT EXAMPLES:
    ✓ SELECT DISTINCT breed_code, size_code FROM dogs
    ✓ SELECT name FROM singer WHERE birth_year = 1948 OR birth_year = 1949
    ✓ SELECT Breeds.breed_name, Sizes.size_description FROM Dogs JOIN Breeds ON Dogs.breed_code = Breeds.breed_code JOIN Sizes ON Dogs.size_code = Sizes.size_code

    WRONG EXAMPLES (DO NOT DO THIS):
    ✗ SELECT b.breed_name FROM breeds b
    ✗ SELECT DISTINCT B.breed_name, S.size_description FROM Dogs D JOIN Breeds B ON D.breed_code = B.breed_code
    ✗ SELECT name FROM singer WHERE birth_year IN (1948, 1949)
    ✗ SELECT T1.name FROM singer AS T1

    Return ONLY the SQL query, no explanations."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                few_shot_prompt,
                ("human", "Question: {question}\nSQL:")
            ])
            
            self.langchain_generator = prompt | llm | StrOutputParser()
            print("✅ Basic LangChain SQL generator initialized")
            
        except Exception as e:
            print(f"⚠️  LangChain setup failed: {e}")
            self.langchain_generator = None

    def _fix_spider_compatibility(self, sql, schema_info):
        """Fix common SQL syntax issues for Spider parser compatibility"""
        if not sql:
            return None
        
        original_sql = sql
        
        # Check for LEFT/RIGHT/OUTER JOIN - Spider doesn't support these well
        if re.search(r'\b(LEFT|RIGHT|OUTER)\s+JOIN\b', sql, re.IGNORECASE):
            print("⚠️ Unsupported JOIN type detected - needs regeneration")
            return None
        
        # Fix: Remove single-letter aliases without AS keyword
        # Pattern: FROM table x or JOIN table x where x is single letter
        sql = re.sub(
            r'\b(FROM|JOIN)\s+(\w+)\s+([a-z])\b',
            r'\1 \2',
            sql,
            flags=re.IGNORECASE
        )
        
        # Fix: Convert IN (val1, val2) to OR conditions for WHERE clauses
        # Only for simple value lists, not subqueries
        in_pattern = r'WHERE\s+(\w+\.?\w*)\s+IN\s+\(([^SELECT][^)]+)\)'
        match = re.search(in_pattern, sql, re.IGNORECASE)
        if match:
            col = match.group(1)
            values = [v.strip() for v in match.group(2).split(',')]
            or_conditions = ' OR '.join([f"{col} = {v}" for v in values])
            sql = re.sub(in_pattern, f'WHERE {or_conditions}', sql, flags=re.IGNORECASE)
            print(f"Fixed IN clause to OR conditions")
        
        # Fix: Standardize aliases to T1, T2, T3 format
        # Find all table aliases
        alias_matches = list(re.finditer(
            r'(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?([A-Z])\b',
            sql,
            re.IGNORECASE
        ))
        
        if alias_matches:
            alias_map = {}
            for i, match in enumerate(alias_matches, 1):
                table_name = match.group(1)
                current_alias = match.group(2)
                
                # Only remap if not already T1, T2, T3, etc.
                if not re.match(r'^T\d+$', current_alias, re.IGNORECASE):
                    new_alias = f'T{i}'
                    alias_map[current_alias] = new_alias
                    print(f"Remapping alias {current_alias} -> {new_alias}")
            
            # Apply alias replacements
            for old_alias, new_alias in alias_map.items():
                # Replace in table declarations
                sql = re.sub(
                    rf'\b{old_alias}\b(?=\s|$|,|\))',
                    new_alias,
                    sql,
                    flags=re.IGNORECASE
                )
            
            # Ensure AS keyword is present
            sql = re.sub(
                r'(FROM|JOIN)\s+(\w+)\s+(T\d+)\b',
                r'\1 \2 AS \3',
                sql,
                flags=re.IGNORECASE
            )
        
        if sql != original_sql:
            print(f"Spider compatibility fixes applied")
            print(f"Fixed SQL: {sql}")
        
        return sql

    def generate_sql_from_question(self, question, db_path):
        """Generate SQL from natural language question using LangChain or patterns"""
        print(f"🔍 Processing question: {question}")
        print(f"📁 Database path: {db_path}")
        
        if not os.path.exists(db_path):
            print(f"❌ Database file not found: {db_path}")
            return "SELECT 1"
        
        schema_info = self._get_db_schema(db_path)
        if not schema_info:
            print("❌ No schema information extracted")
            return "SELECT 1"
        
        # Format schema for prompt
        schema_text = "Available Tables and Columns:\n"
        for table, columns in schema_info.items():
            schema_text += f"Table '{table}': {', '.join(columns)}\n"
        
        if self.langchain_generator:
            try:
                result = self.langchain_generator.invoke({
                    "question": question,
                    "schema": schema_text
                })
                
                cleaned_result = self._clean_sql_result(result)
               
                validated_result = self._validate_and_fix_sql(cleaned_result, schema_info, question)
                
                return validated_result
                #cleaned_sql = self._clean_sql_result(result)
                #print(f"Cleaned SQL: {cleaned_sql}")

                #fixed_sql = self._remove_all_aliases(cleaned_sql, schema_info)
                
                #if fixed_sql is None:
                    #print("🔄 Failed to fix aliases, using fallback...")
                    #return self._pattern_generate_sql(question, schema_info)
                
                #print(f"Final SQL (no aliases): {fixed_sql}")
                #return fixed_sql
                
            except Exception as e:
                print(f"❌ LangChain generation failed: {e}")
        
        return self._pattern_generate_sql(question, schema_info)

    def _remove_all_aliases(self, sql, schema_info):
        """Aggressively remove ALL table aliases"""
        if not sql:
            return None
        
        print(f"🔧 Removing aliases from: {sql}")
        
        # IMPROVED: Match aliases with or without AS, including uppercase like T1, T2
        alias_pattern = r'(?P<keyword>FROM|JOIN)\s+(?P<table>\w+)\s+(?:AS\s+)?(?P<alias>[a-zA-Z]\w*)(?=\s|$|JOIN|WHERE|ON|,)'
        
        matches = list(re.finditer(alias_pattern, sql, re.IGNORECASE))
        
        if not matches:
            print("   No aliases detected")
            return self._normalize_table_case(sql, schema_info)
        
        # Build alias mapping
        alias_to_table = {}
        for match in matches:
            table = match.group('table')
            alias = match.group('alias')
            
            # Map if alias is different from table name
            if alias.lower() != table.lower():
                alias_to_table[alias] = table
                print(f"   Found: {alias} → {table}")
        
        if not alias_to_table:
            return self._normalize_table_case(sql, schema_info)
        
        # Replace alias.column with table.column
        result = sql
        for alias, table in alias_to_table.items():
            result = re.sub(
                rf'\b{re.escape(alias)}\.(\w+)',
                rf'{table}.\1',
                result,
                flags=re.IGNORECASE
            )
        
        # Remove alias declarations - IMPROVED to catch T1, T2, etc
        result = re.sub(
            r'(FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?[a-zA-Z]\w*(?=\s|$|JOIN|WHERE|ON|,)',
            r'\1 \2',
            result,
            flags=re.IGNORECASE
        )
        
        # Normalize table case
        result = self._normalize_table_case(result, schema_info)
        
        print(f"   Result: {result}")
        return result

    def _normalize_table_case(self, sql, schema_info):
        """Normalize table names to match exact case in schema"""
        if not schema_info:
            return sql
        
        result = sql
        table_case_map = {table.lower(): table for table in schema_info.keys()}
        
        for table_lower, table_correct in table_case_map.items():
            # Replace in FROM/JOIN
            result = re.sub(
                rf'\b(FROM|JOIN)\s+{re.escape(table_lower)}\b',
                rf'\1 {table_correct}',
                result,
                flags=re.IGNORECASE
            )
            # Replace in table.column
            result = re.sub(
                rf'\b{re.escape(table_lower)}\.(\w+)',
                rf'{table_correct}.\1',
                result,
                flags=re.IGNORECASE
            )
        
        return result

    def _get_db_schema(self, db_path):
        """Get database schema information"""
        if not os.path.exists(db_path):
            return {}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            schema_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                schema_info[table] = columns
            
            conn.close()
            return schema_info
            
        except Exception as e:
            print(f"Error getting schema: {e}")
            return {}

    def _clean_sql_result(self, result):
        """Clean LLM result to extract SQL and normalize formatting."""
        sql = result.strip()
        
        if sql.lower().startswith("sql:"):
            sql = sql[4:].strip()
        
        if "```sql" in sql.lower():
            start = sql.lower().find("```sql") + 6
            end = sql.find("```", start)
            if end != -1:
                sql = sql[start:end].strip()
        elif "```" in sql:
            start = sql.find("```") + 3
            end = sql.find("```", start)
            if end != -1:
                sql = sql[start:end].strip()
        
        sql = normalize_sql_for_evaluation(sql)
    
        return sql

    def _validate_and_fix_sql(self, sql, schema_info, question):
        """Basic validation and fixing"""
        sql_lower = sql.lower()
        
        from_match = re.search(r'from\s+(\w+)', sql_lower)
        if from_match:
            sql_table = from_match.group(1)
            schema_tables_lower = [t.lower() for t in schema_info.keys()]
            
            if sql_table not in schema_tables_lower:
                correct_table = self._find_best_table_match(question, schema_info)
                sql = re.sub(r'(from\s+)\w+', f'\\1{correct_table}', sql, flags=re.IGNORECASE)
        
        return sql

    def _find_best_table_match(self, question, schema_info):
        """Find the best matching table for the question"""
        question_lower = question.lower()
        tables = list(schema_info.keys())
        
        for table in tables:
            if table.lower() in question_lower:
                return table
        
        return tables[0] if tables else "unknown_table"

    def _pattern_generate_sql(self, question, schema_info):
        """Generate SQL using simple patterns"""
        question_lower = question.lower()
        tables = list(schema_info.keys())
        
        if not tables:
            return "SELECT 1"
        
        primary_table = self._find_best_table_match(question, schema_info)
        
        if re.search(r'how many|count', question_lower):
            return f"SELECT COUNT(*) FROM {primary_table}"
        elif re.search(r'list all|show all', question_lower):
            return f"SELECT * FROM {primary_table}"
        elif re.search(r'names?', question_lower):
            columns = schema_info.get(primary_table, [])
            name_col = next((col for col in columns if 'name' in col.lower()), columns[0] if columns else '*')
            return f"SELECT {name_col} FROM {primary_table}"
        else:
            return f"SELECT * FROM {primary_table}"

    # Keep all your existing evaluation methods
    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def normalize_aliases(sql: str) -> str:
        """Normalize table aliases to T1, T2, T3 format"""
        # Parse SQL and replace aliases with standard format
        # This is a simplified example - you'd need more robust parsing
        
        # Remove alias variations and standardize
        sql = re.sub(r'\bAS\s+', '', sql, flags=re.IGNORECASE)
        
        # Map actual aliases to standard ones
        # You'll need to track which tables get which aliases
        
        return sql

    def eval_exact_match(self, pred, label):
        # Normalize both queries before comparison (if needed)
        # Note: The rebuild functions already do most of the normalization
        pred_normalized = pred  # or self.normalize_sql_structure(pred) if you implement it
        label_normalized = label  # or self.normalize_sql_structure(label) if you implement it

        partial_scores = self.eval_partial_match(pred_normalized, label_normalized)
        self.partial_scores = partial_scores

        for key, score in partial_scores.items():
            if score['f1'] != 1:
                return 0

        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return label_tables == pred_tables
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label, self)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        return res

class ChromaDBEvaluator(BaseEvaluator):
    """Enhanced evaluator with template manager integration AND ChromaDB"""
    
    def __init__(self, prompt_type=PromptType.ENHANCED, enable_debugging=False, 
                 use_chromadb=True, chromadb_config=None):
        super().__init__()
        self.prompt_type = prompt_type
        self.enable_debugging = enable_debugging
        
        # ChromaDB setup - NEW ADDITION
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.chromadb_system = None
        self.retrieval_stats = {
            'queries_with_retrieval': 0,
            'successful_retrievals': 0,
            'retrieval_helped': 0,
            'average_similarity': 0.0
        }
        
        if self.use_chromadb:
            self._setup_chromadb(chromadb_config)
        
        # Initialize template manager if available
        if TEMPLATE_MANAGER_AVAILABLE:
            self.template_manager = TemplateManager()
            self.specialized_prompts = SpecializedPrompts()
            print(f"Using {prompt_type.value} prompting strategy with ChromaDB")
        else:
            self.template_manager = None
            self.specialized_prompts = None
        
        # Override LangChain setup with enhanced templates
        if LANGCHAIN_AVAILABLE and self.template_manager:
            self._setup_enhanced_langchain()
        
        # Statistics tracking
        self.generation_stats = {
            'total_queries': 0,
            'successful_generations': 0,
            'template_corrections': 0,
            'pattern_fallbacks': 0,
            'syntax_errors': 0,
            'chromadb_retrievals': 0
        }

    def _setup_chromadb(self, config=None):
        """Setup ChromaDB system for retrieval - NEW METHOD"""
        try:
            config = config or {}
            data_dir = config.get('data_dir', './spider_data')
            persist_dir = config.get('persist_dir', './chromadb')
            
            self.chromadb_system = InteractiveSpiderQuery(
                data_dir=data_dir,
                persist_dir=persist_dir
            )
            
            # Check if collections are available
            collections_ready = self.chromadb_system.load_or_create_collections()
            if not collections_ready:
                print("ChromaDB collections not ready. Disabling ChromaDB features.")
                self.use_chromadb = False
                self.chromadb_system = None
            else:
                print("ChromaDB system ready for retrieval-augmented generation")
                
        except Exception as e:
            print(f"Failed to setup ChromaDB: {e}")
            self.use_chromadb = False
            self.chromadb_system = None

    def retrieve_similar_examples(self, question: str, n_examples: int = 3, min_similarity: float = 0.3):
        """Retrieve similar examples from ChromaDB for few-shot prompting - NEW METHOD"""
        if not self.use_chromadb or not self.chromadb_system:
            return []
        
        try:
            self.generation_stats['chromadb_retrievals'] += 1
            
            # Get similar questions and SQL queries
            similar_results = self.chromadb_system.query_similar_questions(
                question, n_results=n_examples, min_similarity=min_similarity
            )
            
            if "error" in similar_results:
                return []
            
            # Format examples for prompting
            examples = []
            similarities = []
            
            for result in similar_results.get('results', []):
                examples.append({
                    'question': result['question'],
                    'sql': result['sql_query'],
                    'database': result['database'],
                    'similarity': result['similarity_score']
                })
                similarities.append(result['similarity_score'])
            
            # Update stats
            self.retrieval_stats['successful_retrievals'] += 1
            if similarities:
                self.retrieval_stats['average_similarity'] = sum(similarities) / len(similarities)
            
            return examples
            
        except Exception as e:
            print(f"Error in ChromaDB retrieval: {e}")
            return []

    def retrieve_relevant_schema(self, question: str, n_schemas: int = 2):
        """Retrieve relevant database schemas for the question - NEW METHOD"""
        if not self.use_chromadb or not self.chromadb_system:
            return []
        
        try:
            schema_results = self.chromadb_system.find_relevant_schemas(
                question, n_results=n_schemas
            )
            
            if "error" in schema_results:
                return []
            
            schemas = []
            for schema in schema_results.get('relevant_schemas', []):
                schemas.append({
                    'database': schema['database'],
                    'schema': schema['schema'],
                    'similarity': schema['similarity_score']
                })
            
            return schemas
            
        except Exception as e:
            print(f"Error in schema retrieval: {e}")
            return []

    def _setup_enhanced_langchain(self):
        """Setup enhanced LangChain with template manager AND ChromaDB examples - MODIFIED"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, "..", ".env")
        load_dotenv(env_path)
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your-api-key-here":
            print("Google API key not found. Using pattern matching fallback.")
            return
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
            
            # Get enhanced prompt template from template manager
            template_str = self.template_manager.get_template(self.prompt_type)
            
            # Modify template to include ChromaDB retrieval placeholder - NEW ENHANCEMENT
            if self.use_chromadb:
                chromadb_section = """
## RETRIEVED SIMILAR EXAMPLES:
{retrieved_examples}

Use the above examples as reference for generating accurate SQL queries.
"""
                template_str = template_str.replace(
                    "Database Schema:",
                    chromadb_section + "\nDatabase Schema:"
                )
            
            print(f"Using {self.prompt_type.value} template with ChromaDB enhancement")
            
            # Create LangChain prompt
            prompt = ChatPromptTemplate.from_template(template_str)
            
            # Create chain
            self.langchain_generator = prompt | llm | StrOutputParser()
            print("Enhanced LangChain SQL generator with ChromaDB initialized")
            
        except Exception as e:
            print(f"Enhanced LangChain setup failed: {e}")
            self.langchain_generator = None

    def _remove_all_aliases(self, sql, schema_info):
        """Aggressively remove ALL table aliases"""
        if not sql:
            return None
        
        print(f"🔧 Removing aliases from: {sql}")
        
        # Pattern: FROM/JOIN Table alias (where alias is 1-3 chars or different from table)
        alias_pattern = r'(?P<keyword>FROM|JOIN)\s+(?P<table>\w+)\s+(?:AS\s+)?(?P<alias>[a-zA-Z]+)(?=\s|$|JOIN|WHERE|ON)'
        
        matches = list(re.finditer(alias_pattern, sql, re.IGNORECASE))
        
        if not matches:
            print("   No aliases detected")
            return sql
        
        # Build alias mapping
        alias_to_table = {}
        for match in matches:
            table = match.group('table')
            alias = match.group('alias')
            
            # Map if alias is short (likely an alias) or different from table name
            if len(alias) <= 3 or alias.lower() != table.lower():
                alias_to_table[alias] = table
                print(f"   Found: {alias} → {table}")
        
        if not alias_to_table:
            return sql
        
        # Replace alias.column with Table.column
        result = sql
        for alias, table in alias_to_table.items():
            result = re.sub(
                rf'\b{re.escape(alias)}\.(\w+)',
                rf'{table}.\1',
                result,
                flags=re.IGNORECASE
            )
        
        # Remove alias declarations
        result = re.sub(
            r'(FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?[a-zA-Z]{1,3}(?=\s|$|JOIN|WHERE|ON)',
            r'\1 \2',
            result,
            flags=re.IGNORECASE
        )
        
        print(f"   Result: {result}")
        return result
    
    def generate_sql_from_question(self, question, db_path):
        """Enhanced SQL generation with ChromaDB retrieval and template manager - ENHANCED"""
        self.generation_stats['total_queries'] += 1
        self.retrieval_stats['queries_with_retrieval'] += 1
        
        print(f"Enhanced processing with ChromaDB: {question}")
        print(f"Database: {db_path}")
        
        if not os.path.exists(db_path):
            print(f"Database file not found: {db_path}")
            return "SELECT 1"
        
        schema_info = self._get_db_schema(db_path)
        if not schema_info:
            print("No schema information extracted")
            return "SELECT 1"
        
        print(f"Found {len(schema_info)} tables: {list(schema_info.keys())}")
        
        # Retrieve similar examples from ChromaDB - NEW FUNCTIONALITY
        similar_examples = []
        relevant_schemas = []
        
        if self.use_chromadb:
            similar_examples = self.retrieve_similar_examples(question, n_examples=3)
            relevant_schemas = self.retrieve_relevant_schema(question, n_schemas=2)
            
            if similar_examples:
                print(f"Retrieved {len(similar_examples)} similar examples from ChromaDB")
            if relevant_schemas:
                print(f"Retrieved {len(relevant_schemas)} relevant schemas from ChromaDB")
        
        # Format schema for prompt
        schema_text = self._format_schema_for_prompt(schema_info, relevant_schemas)
        
        # Try LangChain generation with ChromaDB enhancement
        if self.langchain_generator:
            try:
                print(f"Using {self.prompt_type.value} prompting with ChromaDB retrieval...")
                
                # Format retrieved examples for prompt
                retrieved_examples_text = ""
                if similar_examples:
                    retrieved_examples_text = "Similar examples from database:\n"
                    for i, example in enumerate(similar_examples, 1):
                        retrieved_examples_text += f"Example {i} (similarity: {example['similarity']:.3f}):\n"
                        retrieved_examples_text += f"Question: {example['question']}\n"
                        retrieved_examples_text += f"SQL: {example['sql']}\n"
                        retrieved_examples_text += f"Database: {example['database']}\n\n"
                else:
                    retrieved_examples_text = "No similar examples found in database."
                
                result = self.langchain_generator.invoke({
                    "question": question,
                    "schema": schema_text,
                    "retrieved_examples": retrieved_examples_text
                })
                
                print(f"LLM result: {result}")
                cleaned_sql = self._clean_sql_result(result)
                print(f"Cleaned SQL: {cleaned_sql}")
                
                #no_alias_sql = self._remove_all_aliases(cleaned_sql, schema_info)
                
                #if no_alias_sql is None:
                    #print("Failed to process SQL, using fallback")
                    #return self._pattern_generate_sql(question, schema_info)
                
                # Then validate and enhance (uses no_alias_sql, not cleaned_sql)
                #validated_sql = self._validate_and_enhance_sql(no_alias_sql, schema_info, question)
                validated_sql = self._validate_and_enhance_sql(cleaned_sql, schema_info, question)
                
                #if validated_sql != no_alias_sql:
                if validated_sql != cleaned_sql:
                    self.generation_stats['template_corrections'] += 1
                    print(f"Template correction applied: {validated_sql}")
                
                # Check if retrieval helped
                if similar_examples and self._sql_seems_better_with_retrieval(validated_sql, similar_examples):
                    self.retrieval_stats['retrieval_helped'] += 1
                
                self.generation_stats['successful_generations'] += 1
                return validated_sql
                
            except Exception as e:
                print(f"Enhanced LangChain generation failed: {e}")
                self.generation_stats['syntax_errors'] += 1
                if self.enable_debugging:
                    import traceback
                    traceback.print_exc()

        # Fallback to enhanced pattern matching with ChromaDB
        print("Using enhanced pattern matching with ChromaDB fallback...")
        self.generation_stats['pattern_fallbacks'] += 1
        return self._enhanced_pattern_generate_sql_with_chromadb(question, schema_info, similar_examples)

    def _format_schema_for_prompt(self, schema_info, relevant_schemas=None):
        """Format schema information for prompt templates with ChromaDB context"""
        if not schema_info:
            return "No schema available"
        
        schema_lines = []
        
        # Add main schema
        schema_lines.append("=== Current Database Schema ===")
        for table, columns in schema_info.items():
            schema_lines.append(f"table: {table} [{', '.join(columns)}]")
        
        # Add relevant schemas as context if available
        if relevant_schemas:
            schema_lines.append("\n=== Similar Database Schemas for Context ===")
            for i, rel_schema in enumerate(relevant_schemas, 1):
                schema_lines.append(f"Example {i} (similarity: {rel_schema['similarity']:.3f}):")
                schema_lines.append(rel_schema['schema'])
                schema_lines.append("")
        
        return "\n".join(schema_lines)

    def _validate_and_enhance_sql(self, sql, schema_info, question):
        """Enhanced validation using template manager patterns"""
        if not sql or sql == "SELECT 1":
            return sql
        
        # Basic validation
        issues = self._find_sql_issues(sql, schema_info)
        
        if not issues:
            return sql
        
        print(f"Found {len(issues)} validation issues: {[i['type'] for i in issues]}")
        
        # Try to fix using specialized prompts if available
        if self.specialized_prompts and self.langchain_generator and self.enable_debugging:
            try:
                debug_template = self.specialized_prompts.get_debugging_prompt()
                
                # Create debugging chain
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    convert_system_message_to_human=True
                )
                
                debug_prompt = ChatPromptTemplate.from_template(debug_template)
                debug_chain = debug_prompt | llm | StrOutputParser()
                
                schema_text = self._format_schema_for_prompt(schema_info)
                error_msg = "; ".join([f"{i['type']}: {i.get('message', 'Issue detected')}" for i in issues])
                
                corrected_result = debug_chain.invoke({
                    "question": question,
                    "schema": schema_text,
                    "sql": sql,
                    "error": error_msg
                })
                
                corrected_sql = self._clean_sql_result(corrected_result)
                print(f"Debugging prompt result: {corrected_sql}")
                
                return corrected_sql
                
            except Exception as e:
                print(f"Debugging prompt failed: {e}")
        
        # Fallback to rule-based corrections
        return self._apply_rule_based_corrections(sql, issues, schema_info, question)

    def _find_sql_issues(self, sql, schema_info):
        """Find issues in the generated SQL"""
        issues = []
        sql_lower = sql.lower()
        
        # Check table existence
        from_match = re.search(r'from\s+(\w+)', sql_lower)
        if from_match:
            table_name = from_match.group(1)
            if table_name not in [t.lower() for t in schema_info.keys()]:
                issues.append({
                    'type': 'table_not_found',
                    'table': table_name,
                    'message': f"Table '{table_name}' not found in schema"
                })
        
        return issues

    def _apply_rule_based_corrections(self, sql, issues, schema_info, question):
        """Apply rule-based corrections using template manager patterns"""
        corrected_sql = sql
        
        for issue in issues:
            if issue['type'] == 'table_not_found':
                correct_table = self._find_best_table_match(question, schema_info)
                pattern = r'(from\s+)\w+'
                corrected_sql = re.sub(pattern, f'\\1{correct_table}', corrected_sql, flags=re.IGNORECASE)
                print(f"Corrected table: {issue['table']} → {correct_table}")
        
        return corrected_sql

    def _sql_seems_better_with_retrieval(self, sql: str, similar_examples) -> bool:
        """Check if SQL generation was improved by ChromaDB retrieval - NEW METHOD"""
        # Simple heuristic: check if generated SQL has patterns similar to retrieved examples
        sql_lower = sql.lower()
        
        for example in similar_examples:
            example_sql = example['sql'].lower()
            
            # Check for similar SQL patterns
            if any(pattern in sql_lower for pattern in ['select', 'from', 'where', 'join']):
                if any(pattern in example_sql for pattern in ['select', 'from', 'where', 'join']):
                    return True
        
        return False

    def _enhanced_pattern_generate_sql_with_chromadb(self, question, schema_info, similar_examples):
        """Enhanced pattern-based SQL generation using ChromaDB examples - NEW METHOD"""
        question_lower = question.lower()
        tables = list(schema_info.keys())
        
        if not tables:
            return "SELECT 1"
        
        # First, try to learn from ChromaDB examples
        if similar_examples:
            for example in similar_examples:
                example_sql = example['sql'].lower()
                example_question = example['question'].lower()
                
                # Pattern matching from retrieved examples
                if ('count' in question_lower or 'how many' in question_lower) and 'count' in example_sql:
                    primary_table = self._find_best_table_match(question, schema_info)
                    return f"SELECT COUNT(*) FROM {primary_table}"
                
                if ('list' in question_lower or 'show' in question_lower) and 'select *' in example_sql:
                    primary_table = self._find_best_table_match(question, schema_info)
                    return f"SELECT * FROM {primary_table}"
                
                if 'join' in example_sql and any(word in question_lower for word in ['with', 'and', 'together']):
                    # Try to replicate JOIN pattern
                    if len(tables) > 1:
                        return f"SELECT * FROM {tables[0]} JOIN {tables[1]}"
        
        # Fallback to template manager patterns
        if self.template_manager:
            patterns = self.template_manager.common_patterns
            
            for pattern, sql_template in patterns.items():
                if pattern.lower() in question_lower:
                    print(f"Matched pattern: '{pattern}' → {sql_template}")
                    customized_sql = self._customize_sql_template(sql_template, question, schema_info)
                    return customized_sql
        
        # Final fallback to basic pattern matching
        return super()._pattern_generate_sql(question, schema_info)

    def _customize_sql_template(self, template, question, schema_info):
        """Customize SQL template based on question and schema"""
        primary_table = self._find_best_table_match(question, schema_info)
        customized = template.replace('table', primary_table)
        
        if 'column' in customized:
            relevant_column = self._find_relevant_column(question, primary_table, schema_info)
            customized = customized.replace('column', relevant_column)
        
        return customized

    def _find_relevant_column(self, question, table, schema_info):
        """Find the most relevant column based on question context"""
        if table not in schema_info:
            return '*'
        
        question_lower = question.lower()
        columns = schema_info[table]
        
        # Look for columns mentioned in question
        for col in columns:
            if col.lower() in question_lower:
                return col
        
        # Look for name columns
        for col in columns:
            if 'name' in col.lower():
                return col
        
        return columns[0] if columns else '*'

    def get_generation_statistics(self):
        """Get comprehensive statistics including ChromaDB performance - ENHANCED"""
        total = self.generation_stats['total_queries']
        if total == 0:
            return {}
        
        stats = {
            'total_queries': total,
            'success_rate': self.generation_stats['successful_generations'] / total,
            'correction_rate': self.generation_stats['template_corrections'] / total,
            'fallback_rate': self.generation_stats['pattern_fallbacks'] / total,
            'error_rate': self.generation_stats['syntax_errors'] / total,
        }
        
        stats.update({
            'chromadb_retrievals': self.generation_stats['chromadb_retrievals'],
            'retrieval_success_rate': (
                self.retrieval_stats['successful_retrievals'] / 
                max(1, self.retrieval_stats['queries_with_retrieval'])
            ),
            'retrieval_help_rate': (
                self.retrieval_stats['retrieval_helped'] / 
                max(1, self.retrieval_stats['queries_with_retrieval'])
            ),
            'average_similarity': self.retrieval_stats['average_similarity']
        })
        
        return stats

    def get_retrieval_statistics(self):
        """Get detailed ChromaDB retrieval statistics"""
        stats = self.retrieval_stats.copy()
        
        if stats['queries_with_retrieval'] > 0:
            stats['retrieval_success_rate'] = stats['successful_retrievals'] / stats['queries_with_retrieval']
            stats['retrieval_help_rate'] = stats['retrieval_helped'] / stats['queries_with_retrieval']
        else:
            stats['retrieval_success_rate'] = 0.0
            stats['retrieval_help_rate'] = 0.0
        
        return stats
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str, help="the path to the gold queries")
    parser.add_argument('--pred', dest='pred', type=str, help="the path to the predicted queries")
    parser.add_argument('--db', dest='db', type=str, help="the directory that contains all the databases and test suites")
    parser.add_argument('--table', dest='table', type=str, help="the tables.json schema file")
    parser.add_argument('--etype', dest='etype', type=str, default='exec',
                        help="evaluation type, exec for test suite accuracy, match for the original exact set match accuracy",
                        choices=('all', 'exec', 'match'))
    parser.add_argument('--plug_value', default=False, action='store_true',
                        help='whether to plug in the gold value into the predicted query; suitable if your model does not predict values.')
    parser.add_argument('--keep_distinct', default=False, action='store_true',
                        help='whether to keep distinct keyword during evaluation. default is false.')
    parser.add_argument('--progress_bar_for_each_datapoint', default=False, action='store_true',
                        help='whether to print progress bar of running test inputs for each datapoint')
    parser.add_argument('--use_langchain', default=False, action='store_true',
                        help='Use LangChain to generate SQL from natural language questions')
    parser.add_argument('--questions', dest='questions', type=str, 
                        help='Path to file containing natural language questions (required when using --use_langchain)')
    parser.add_argument('--prompt_type', dest='prompt_type', type=str, default='enhanced',
                        help='Type of prompting strategy to use',
                        choices=['basic', 'few_shot', 'chain_of_thought', 'rule_based', 'enhanced', 'step_by_step'])
    parser.add_argument('--enable_debugging', default=False, action='store_true',
                        help='Enable debugging prompts for SQL correction')
    parser.add_argument('--use_enhanced', default=False, action='store_true',
                        help='Use enhanced evaluator with template manager')
    
    # ChromaDB options
    parser.add_argument('--use_chromadb', default=False, action='store_true',
                        help='Enable ChromaDB retrieval-augmented generation')
    parser.add_argument('--chromadb_data_dir', dest='chromadb_data_dir', type=str, default='./spider_data',
                        help='Path to Spider dataset directory for ChromaDB')
    parser.add_argument('--chromadb_persist_dir', dest='chromadb_persist_dir', type=str, default='./chromadb',
                        help='Path to ChromaDB persistence directory')
    parser.add_argument('--chromadb_n_examples', dest='chromadb_n_examples', type=int, default=3,
                        help='Number of similar examples to retrieve from ChromaDB')
    parser.add_argument('--chromadb_min_similarity', dest='chromadb_min_similarity', type=float, default=0.3,
                        help='Minimum similarity threshold for ChromaDB retrieval')
    parser.add_argument('--chromadb_n_schemas', dest='chromadb_n_schemas', type=int, default=2,
                        help='Number of relevant schemas to retrieve from ChromaDB')
    parser.add_argument('--use_semantic', default=False, action='store_true',
                       help='Use semantic layer for enhanced SQL generation')
    
    args = parser.parse_args()

    # Prepare ChromaDB configuration
    chromadb_config = None
    if args.use_chromadb:
        if not CHROMADB_AVAILABLE:
            print("❌ ChromaDB not available. Please run the setup script first:")
            print("  python utils/spider_chromadb_integration.py")
            print("Or disable ChromaDB with --use_chromadb=False")
            exit(1)
        
        chromadb_config = {
            'data_dir': args.chromadb_data_dir,
            'persist_dir': args.chromadb_persist_dir,
            'n_examples': args.chromadb_n_examples,
            'min_similarity': args.chromadb_min_similarity,
            'n_schemas': args.chromadb_n_schemas
        }
        
        print(f"🔍 ChromaDB enabled with config: {chromadb_config}")

    # only evaluting exact match needs this argument
    kmaps = None
    if args.etype in ['all', 'match']:
        assert args.table is not None, 'table argument must be non-None if exact set match is evaluated'
        kmaps = build_foreign_key_map_from_json(args.table)

    results = evaluate(
        args.gold, args.pred, args.db, args.etype, kmaps, args.plug_value, 
        args.keep_distinct, args.progress_bar_for_each_datapoint, 
        args.use_langchain, args.questions, args.prompt_type, args.enable_debugging,
        args.use_chromadb, chromadb_config, args.use_semantic
    )
    
    print("Evaluation completed!")
    print(f"Results: {results}")