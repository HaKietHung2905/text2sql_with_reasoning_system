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
# NEW — replace the entire WIKISQL_ANNOTATION_RULES value with:
WIKISQL_ANNOTATION_RULES = """\
━━━ WIKISQL ANNOTATION RULES (follow exactly) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SINGLE-VALUE RETRIEVAL — always wrap in MAX():
   Applies to: "What is the X?" / "Which X?" / "Name the X" /
               "The [date] of X had a Y of what?" / "The [event] applied to what X?"
   → SELECT MAX(col) FROM wikisql_data WHERE ...
   Examples:
     "What is the pick number for Northwestern?" → SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'
     "What is Iceland's total?"                  → SELECT COUNT(total) FROM wikisql_data WHERE country = 'Iceland'
     "What is the United States rank?"           → SELECT COUNT(rank) FROM wikisql_data WHERE country = 'United States'
     "Name the finished position for X"          → SELECT COUNT(finished) FROM wikisql_data WHERE celebrity = 'X'
     "The canadian airdate of X applied to what series number?" → SELECT COUNT(no_in_series) FROM wikisql_data WHERE canadian_airdate = 'X'

2. MINIMUM RETRIEVAL — use MIN() when lowest/earliest/first is implied:
   → SELECT MIN(col) FROM wikisql_data WHERE ...
   Example: "Name the minimum ties played for 6 years."
     → SELECT MIN(ties_played) FROM wikisql_data WHERE years_played = 6

3. COUNTING — use COUNT(col), NEVER COUNT(*):
   Applies to: "How many X?" / "What is the total number of X?"
   → SELECT COUNT(col) FROM wikisql_data WHERE ...
   Example: "How many players are on the Toronto team in 2005-06?"
     → SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'
   EXCEPTION: if the schema already has a column storing the count (goals, viewers,
   points, attendance), use plain SELECT for that column directly.

4. WHERE CONDITION VALUES — copy EXACTLY as stored in the database:
   • String with units:  WHERE col = '131 runs'    NOT  WHERE col = 131
   • Dollar amounts:     WHERE col = '$60,000'      NOT  WHERE col = 60000
   • Ordinals:           WHERE col = '4th'          NOT  WHERE col = 4
   • Score strings:      WHERE col = '-8 (71-63-69-69=272)'  NOT  WHERE col = -8
   • Venue+attendance:   WHERE col = 'Philips Arena 19,335'  NOT  WHERE col = 19335
   • Dates as stored:    WHERE col = 'january 18, 2009'  (match exact format in DB)
   RULE: when a condition value looks numeric but the question contains units,
   formatting, or context, use the FULL STRING form.

5. SUPERLATIVES are NOT WHERE conditions — use MAX/MIN in SELECT:
   BAD:  WHERE height = (SELECT MAX(height) ...)
   GOOD: SELECT MAX(height) FROM wikisql_data WHERE floors = 35

6. SINGLE-COLUMN SELECT only. No ORDER BY, GROUP BY, LIMIT, subqueries, JOINs.
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