"""
WikiSQL Prompt Patch — Annotation Convention Fix
=================================================
WikiSQL annotators followed a specific convention when writing gold SQL
that differs from natural SQL:

  QUIRK 1 — MAX/MIN for single-row retrieval (79 cases failing)
    "What is the episode number for code 8ABX15?"
    Gold:  SELECT MAX(col1) FROM table WHERE col6 = '8ABX15'
    Model: SELECT no_in_series FROM wikisql_data WHERE ...    ← loses 79 EM

  QUIRK 2 — COUNT(col) not COUNT(*) (48 cases failing)
    Gold:  SELECT COUNT(col3) FROM table WHERE col4 = '2005-06'
    Model: SELECT COUNT(*) FROM wikisql_data WHERE ...        ← evaluator now
                                                                handles this ✓

SOLUTION
--------
Add a WikiSQL-specific annotation block to the prompt in
src/evaluation/sql_generator.py (or your prompt builder) when
is_wikisql=True. This block must come RIGHT BEFORE the question.

HOW TO APPLY
------------
Find where you build the WikiSQL prompt (likely in _build_prompt or
the WikiSQL-specific template in improved_prompt.py). Add the
WIKISQL_ANNOTATION_RULES block shown below.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Drop this constant into src/evaluation/sql_generator.py
# or wherever your WikiSQL prompt is assembled.
# ══════════════════════════════════════════════════════════════════════════════
WIKISQL_ANNOTATION_RULES = """\
━━━ WIKISQL ANNOTATION RULES (follow exactly) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SINGLE-VALUE RETRIEVAL — always wrap in MAX():
   • "What is the X?" / "Which X?" / "Name the X"
   • 'The [date/event] of X applied to what Y?' → MAX or COUNT
   • 'The [date/event] of X had a Y of what?' → MAX
   → SELECT MAX(col) FROM wikisql_data WHERE ...

2. MINIMUM RETRIEVAL — use MIN() when lowest/earliest/first is implied:
   • "What is the lowest/first/earliest X?" / "Name the minimum X"
   → SELECT MIN(col) FROM wikisql_data WHERE ...
   • Example: "Name the minimum ties played for 6 years."
     → SELECT MIN(ties_played) FROM wikisql_data WHERE years_played = 6

3. COUNTING — always use COUNT(col), never COUNT(*):
   • "How many X?" / "What is the total number of X?"
   → SELECT COUNT(col) FROM wikisql_data WHERE ...
   • Use the column being counted, not *.
   • Example: "How many players played in 2005-06?"
     → SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'

3b. EXCEPTION — plain SELECT when a schema column already holds the count:
   • If a column named goals/viewers/points/seats/attendance exists in the
     schema and the question asks "how many [that noun]", use plain SELECT.
   • "How many goals were scored in 2005-06?"
     → SELECT goals FROM wikisql_data WHERE season = '2005-06'
   Rule: check the schema first — if the noun maps to a column name, SELECT it.
3c. COUNT vs SUM — critical distinction:
   • 'total number of X' → COUNT(col)  ← ALWAYS, even if X sounds numeric
   • 'total X' (bare)    → SUM(col)
   • 'how many X'        → COUNT(col)
   • Example: 'total number of episode count' → COUNT(final_episode_count)
   • Example: 'total attendance'              → SUM(attendance)

4. WHERE conditions — include ALL filters explicitly stated, nothing more:
   • Add a condition for EVERY filter criterion named in the question.
   • Do NOT invent, infer, or add conditions not present in the question.
   • SUPERLATIVE RULE: words like "tallest", "largest", "most recent" are NOT
     WHERE conditions — represent them as MAX/MIN in the SELECT clause instead.
     WRONG: WHERE height = (SELECT MAX(height) FROM wikisql_data)
     RIGHT:  SELECT MAX(height) FROM wikisql_data WHERE floors = 35
   • No subqueries, nested SELECTs, or (SELECT ...) anywhere in the query.

5. COMPOUND WHERE VALUES — never split on commas:
   • WHERE regular_season = '4th, Atlantic Division'  ← correct
   • WHERE regular_season = '4th' AND ...             ← wrong

6. String values: always quote with single quotes.
   Numeric values: do NOT quote.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ══════════════════════════════════════════════════════════════════════════════
# Updated WikiSQL prompt template — replace IMPROVED_PROMPT_TEMPLATE_WIKISQL
# in improved_prompt.py with this version.
# ══════════════════════════════════════════════════════════════════════════════

WIKISQL_PROMPT_TEMPLATE_V2 = """\
You are a Text-to-SQL expert for WikiSQL.

━━━ OUTPUT RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Output ONE SQL SELECT query only. No explanations. No markdown. No semicolon.
• Table name is always: wikisql_data
• Use EXACTLY the column names listed in the schema below (case-preserved).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{annotation_rules}

Table: wikisql_data
Columns: {columns}

{few_shot_block}\
Question: {question}

SQL:"""


# ══════════════════════════════════════════════════════════════════════════════
# Wire-up in sql_generator.py — replace the wikisql branch of _build_prompt:
# ══════════════════════════════════════════════════════════════════════════════
#
#   if is_wikisql:
#       from src.evaluation.wikisql_prompt_patch import (
#           WIKISQL_PROMPT_TEMPLATE_V2, WIKISQL_ANNOTATION_RULES
#       )
#       columns = schema_text   # pass column list string directly
#       return WIKISQL_PROMPT_TEMPLATE_V2.format(
#           annotation_rules = WIKISQL_ANNOTATION_RULES,
#           columns          = columns,
#           question         = question,
#           few_shot_block   = few_shot_block,
#       )
#
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# Quick validation — test that the annotation rules recover known failure cases
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_QUIRK_EXAMPLES = [
    {
        "question":  "What is the episode number that has production code 8abx15?",
        "gold_sql":  "SELECT MIN(col1) FROM table WHERE col6 = '8ABX15'",
        "bad_pred":  "SELECT no_in_series FROM wikisql_data WHERE production_code = '8ABX15'",
        "good_pred": "SELECT MIN(no_in_series) FROM wikisql_data WHERE production_code = '8ABX15'",
        "rule":      "QUIRK 1 MIN — single-value retrieval needs MIN()",
    },
    {
        "question":  "What is the pick number for Northwestern college?",
        "gold_sql":  "SELECT MAX(col0) FROM table WHERE col4 = 'Northwestern'",
        "bad_pred":  "SELECT pick FROM wikisql_data WHERE college = 'Northwestern'",
        "good_pred": "SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'",
        "rule":      "QUIRK 1 MAX — single-value retrieval needs MAX()",
    },
    {
        "question":  "How many players are on the Toronto team in 2005-06?",
        "gold_sql":  "SELECT COUNT(col1) FROM table WHERE col4 = '2005-06'",
        "bad_pred":  "SELECT COUNT(*) FROM wikisql_data WHERE years_in_toronto = '2005-06'",
        "good_pred": "SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'",
        "rule":      "QUIRK 2 COUNT — use COUNT(col) not COUNT(*)",
    },
]

if __name__ == "__main__":
    print("WikiSQL annotation quirk examples:\n")
    for ex in KNOWN_QUIRK_EXAMPLES:
        print(f"  Rule    : {ex['rule']}")
        print(f"  Question: {ex['question']}")
        print(f"  Gold    : {ex['gold_sql']}")
        print(f"  Bad pred: {ex['bad_pred']}")
        print(f"  Fixed   : {ex['good_pred']}")
        print()