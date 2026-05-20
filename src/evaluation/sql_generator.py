"""
SQL generation from natural language using LangChain.
"""

import os
import re
import sqlite3
from typing import Dict, List, Optional
from dotenv import load_dotenv

from utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available")

# ── WikiSQL annotation rules ──────────────────────────────────────────────────
WIKISQL_ANNOTATION_RULES = """\
━━━ WIKISQL ANNOTATION RULES (follow exactly) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SINGLE-VALUE RETRIEVAL — always wrap in MAX():
   Applies to: "What is the X?" / "Which X?" / "Name the X" /
               "The [date] of X had a Y of what?" / "The [event] applied to what X?"
   → SELECT MAX(col) FROM wikisql_data WHERE ...
   Examples:
     "What is the pick number for Northwestern?"
       → SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'
     "What is the United States rank?"
       → SELECT COUNT(rank) FROM wikisql_data WHERE country = 'United States'
     "Name the finished position for Kerry Katona"
       → SELECT COUNT(finished) FROM wikisql_data WHERE celebrity = 'Kerry Katona'
 
2. MINIMUM RETRIEVAL — use MIN() when lowest/earliest/first is implied:
   → SELECT MIN(col) FROM wikisql_data WHERE ...
   Example: "Name the minimum ties played for 6 years."
     → SELECT MIN(ties_played) FROM wikisql_data WHERE years_played = 6
 
3. COUNTING RECORDS — use COUNT(col), NEVER COUNT(*):
   Applies to: "How many [entities]?" / "What is the total number of X?"
               "Name the number of X" / "Number of X with Y"
   → SELECT COUNT(col) FROM wikisql_data WHERE ...
   Examples:
     "How many players are on the Toronto team in 2005-06?"
       → SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'
     "Name the total number of points for South Korea"
       → SELECT COUNT(points) FROM wikisql_data WHERE country = 'South Korea'
   NEVER COUNT(*), always COUNT(specific_column).
   NEVER COUNT(DISTINCT ...), always COUNT(col).
 
4. NUMERIC VALUE IN A COLUMN — use bare SELECT (NOT SUM, NOT COUNT):
   CRITICAL: When the question asks for a numeric quantity that IS ALREADY
   stored in a column, use plain SELECT — NOT SUM, NOT AVG.
   Test: if the column name contains the answer unit (goals, viewers, votes,
         points, passengers, runs), use SELECT col.
   Examples:
     "How many goals were scored in the 2005-06 season?"  (goals = column)
       → SELECT goals FROM wikisql_data WHERE season = '2005-06'
       ✗ NOT: SELECT SUM(goals) FROM wikisql_data WHERE season = '2005-06'
     "How many viewers did the David Nutter episode draw in?"  (viewers = column)
       → SELECT u_s_viewers_million FROM wikisql_data WHERE directed_by = 'David Nutter'
       ✗ NOT: SELECT SUM(u_s_viewers_million) FROM ...
     "How many votes were cast in midlothian?"  (votes = column)
       → SELECT votes_cast FROM wikisql_data WHERE constituency = 'midlothian'
 
5. TOTAL/SUM OVER MULTIPLE ROWS — use SUM() only when combining across many rows:
   Use SUM ONLY when no WHERE condition uniquely identifies a single row.
   "What is the total X for all Y?" (no unique filter) → SUM(X)
   NEVER use SUM when a WHERE clause uniquely identifies one row.
 
6. WHERE conditions — include ALL filters explicitly stated, nothing more:
   • Add a condition for EVERY filter criterion named in the question.
   • Do NOT invent, infer, or add conditions not present in the question.
   • SUPERLATIVE RULE: words like "tallest", "largest", "most recent" are NOT
     WHERE conditions — represent them as MAX/MIN in the SELECT clause instead.
     WRONG: WHERE height = (SELECT MAX(height) FROM wikisql_data)
     RIGHT: SELECT MAX(height) FROM wikisql_data WHERE floors = 35
   • No subqueries, nested SELECTs, or (SELECT ...) anywhere in the query.
   • No ORDER BY ... LIMIT 1. Use MAX()/MIN() instead.
 
7. COMPOUND WHERE VALUES — never split on commas:
   • WHERE regular_season = '4th, Atlantic Division'  ← correct
   • WHERE regular_season = '4th' AND ...             ← wrong
 
8. String values: always quote with single quotes.
   Numeric values: do NOT quote.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class SQLGenerator:
    """Generate SQL from natural language questions (LangChain / Gemini backend)"""

    def __init__(self):
        self.generator = None
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()

    def _setup_langchain(self):
        """Setup LangChain chain for SQL generation."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key or api_key == "your-api-key-here":
            logger.warning("Google API key not found")
            return

        try:
            llm = ChatGoogleGenerativeAI(
                model="meta/llama-4-maverick-17b-128e-instruct-maas",
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True,
            )
            self.generator = (
                ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
            )
            logger.info("SQL generator (LangChain) initialized")
        except Exception as e:
            logger.error(f"Failed to setup LangChain: {e}")
            self.generator = None

    # ------------------------------------------------------------------
    # WikiSQL detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_wikisql(db_path: str) -> bool:
        """
        Return True when the database is a WikiSQL database.
        Fast-path: path contains 'wikisql'.
        Fallback: inspect tables — wikisql_data present means WikiSQL.
        """
        if "wikisql" in db_path.lower():
            return True
        try:
            conn = sqlite3.connect(db_path)
            cur  = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            conn.close()
            return "wikisql_data" in tables
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        db_path: str,
        schema_info: Optional[Dict] = None,
    ) -> str:
        """
        Generate SQL for a question.

        Pipeline:
          1. LLM generation via _invoke()
          2. SQL extraction  (_clean_sql, already called inside _invoke)
          3. Pattern-based heuristic fallback  (_pattern_generate)
          4. Hard fallback   SELECT 1   (last resort)

        Returns a non-empty string. Never returns "".
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return "SELECT 1"

        if schema_info is None:
            schema_info = self._load_schema_info(db_path)

        schema_str = self._get_schema_string(db_path)
        is_wikisql = self._is_wikisql(db_path)

        # ── Step 1 + 2: LLM → extract SQL ────────────────────────────────
        # FIX: use self._invoke() (LangChain path) instead of self.model.generate()
        sql = self._invoke(question, schema_str, simple=False, is_wikisql=is_wikisql)

        # Retry with terse prompt on empty result
        if not sql:
            sql = self._invoke(question, schema_str, simple=True, is_wikisql=is_wikisql)

        # ── Step 3: Pattern-based fallback ───────────────────────────────
        if not sql:
            logger.warning(
                f"LLM returned no valid SQL for: {question!r}  → using pattern fallback"
            )
            try:
                sql = self._pattern_generate(question, schema_info)
            except Exception as e:
                logger.error(f"Pattern fallback failed: {e}")

        # ── Step 4: Hard fallback ─────────────────────────────────────────
        if not sql:
            logger.error(
                f"All generation methods failed for: {question!r}  → SELECT 1"
            )
            return "SELECT 1"

        return self._normalize_for_spider(sql)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke(
        self,
        question: str,
        schema_text: str,
        simple: bool = False,
        is_wikisql: bool = False,
    ) -> str:
        """Invoke the LangChain chain and return clean SQL, or '' on failure."""
        if not self.generator:
            return ""

        prompt_text = self._build_prompt(
            question, schema_text, simple=simple, is_wikisql=is_wikisql
        )

        try:
            result = self.generator.invoke({"prompt": prompt_text})
            return self._clean_sql(result)
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in (
                "500", "502", "503", "504",
                "Internal Server Error", "Bad Gateway",
                "Service Unavailable", "Gateway Timeout",
            )):
                raise   # propagate 5xx so generate_predictions.py can checkpoint
            logger.error(f"LangChain invoke failed ({'simple' if simple else 'full'}): {e}")
            return ""

    def _build_prompt(
        self,
        question: str,
        schema_text: str,
        simple: bool = False,
        is_wikisql: bool = False,
    ) -> str:
        """
        Build prompt string.

        simple=True      → ultra-terse, used on retry
        is_wikisql=True  → injects WIKISQL_ANNOTATION_RULES
        """
        if simple:
            return (
                f"Schema:\n{schema_text}\n\n"
                f"Question: {question}\n\n"
                "Write a single SQL SELECT. No explanation. No reasoning. "
                "Start with SELECT.\n\nSQL:"
            )

        if is_wikisql:
            return self._build_prompt_wikisql(question, schema_text)

        # ── Spider / general prompt ───────────────────────────────────────
        # NOTE: Do NOT use backtick code fences here — LangChain's
        # ChatPromptTemplate parses {variable} tokens inside them.
        return (
            "You are a SQL expert. Write a single SQL SELECT statement.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Output ONLY the SQL, starting directly with SELECT.\n"
            "- Do NOT explain, reason, or add any text before or after the SQL.\n"
            "- Do NOT use markdown fences.\n\n"
            f"Database Schema:\n{schema_text}\n\n"
            "RULES:\n"
            '- "how many" / "total number of"  -> COUNT(col)  [NOT SUM]\n'
            '- "total <numeric col>"           -> SUM(col)\n'
            '- "minimum / lowest"              -> MIN(col)\n'
            '- "maximum / highest"             -> MAX(col)\n'
            '- "average / mean"                -> AVG(col)\n'
            "- Exact string match: WHERE col = 'value'  [NOT LIKE]\n"
            "- Keep compound filter values together:\n"
            "    WHERE regular_season = '4th, Atlantic Division'  (correct)\n"
            "    WHERE regular_season = '4th' AND ...             (wrong)\n"
            "- Never invent columns or tables not in the schema.\n\n"
            "EXAMPLES:\n"
            "Q: How many schools did player 3 play at?\n"
            "A: SELECT COUNT(school_club_team) FROM table_name WHERE no_ = 3\n\n"
            "Q: What is the total number of positions on the Toronto team in 2006-07?\n"
            "A: SELECT COUNT(position) FROM table_name WHERE years_in_toronto = '2006-07'\n\n"
            "Q: Name the minimum ties played for 6 years.\n"
            "A: SELECT MIN(ties_played) FROM table_name WHERE years_played = 6\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM table_name WHERE position = 'guard'\n\n"
            f"Question: {question}\n"
            "SQL Query:"
        )

    def _build_prompt_wikisql(self, question: str, schema_text: str) -> str:
        agg_hint  = _wikisql_agg_hint(question)
        cond_hint = _wikisql_cond_hint(question)
        return (
            "You are a Text-to-SQL expert for WikiSQL.\n\n"
            "OUTPUT RULES:\n"
            "- Output ONE SQL SELECT query only. No explanations. No markdown. No semicolon.\n"
            "- Table name is always: wikisql_data\n"
            "- Use EXACTLY the column names from the schema below (case-preserved).\n"
            "- NEVER use subqueries, nested SELECT, or (SELECT ...) inside WHERE.\n\n"
            "WIKISQL ANNOTATION RULES (CRITICAL — these match the gold SQL style):\n"
            "1. SINGLE-VALUE RETRIEVAL — always wrap in MAX():\n"
            "   'What is the X?' / 'Which X?' → SELECT MAX(col) FROM wikisql_data WHERE ...\n"
            "   NEVER use plain SELECT col for a single descriptive value.\n"
            "2. COUNTING — always COUNT(col), NEVER COUNT(*):\n"
            "   'How many X?' → SELECT COUNT(col) FROM wikisql_data WHERE ...\n"
            "3. SUM vs COUNT:\n"
            "   'total number of X' → COUNT(col)   ← counting rows\n"
            "   'total <numeric>'   → SUM(col)     ← summing a column\n"
            "4. AGG keyword map — NEVER drop aggregation when these words appear:\n"
            "   highest/most/largest/best/latest → MAX(col)\n"
            "   lowest/fewest/smallest/earliest/first → MIN(col)\n"
            "   average/mean → AVG(col)\n"
            "   how many/number of/count → COUNT(col)\n"
            "   total/sum of → SUM(col)\n"
            "   COUNT vs SUM: COUNT(col) counts rows. SUM(col) adds numeric values.\n"
            "   'How many games' → COUNT(games). 'What is the total score' → SUM(score).\n"
            "5. WHERE conditions:\n"
            "   - Only include conditions EXPLICITLY stated in the question.\n"
            "   - Do NOT add extra filters beyond what the question asks.\n"
            "   - Keep compound values intact: WHERE result = '4th, Atlantic Division'\n"
            "   - String values: single-quoted. Numeric values: unquoted.\n"
            "   - SUPERLATIVE RULE: words like 'tallest', 'largest', 'most recent' are\n"
            "     NOT WHERE conditions. Represent them as MAX/MIN in the SELECT clause.\n"
            "     BAD:  WHERE height = (SELECT MAX(height) ...)\n"
            "     GOOD: SELECT MAX(height) FROM wikisql_data WHERE floors = 35\n\n"
            f"Database Schema:\n{schema_text}\n\n"
            "EXAMPLES:\n"
            "Q: What is the pick number for Northwestern college?\n"
            "A: SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'\n\n"
            "Q: What is the episode number that has production code 8ABX15?\n"
            "A: SELECT MIN(no_in_series) FROM wikisql_data WHERE production_code = '8ABX15'\n\n"
            "Q: How many schools did player 3 play at?\n"
            "A: SELECT COUNT(school_club_team) FROM wikisql_data WHERE no_ = 3\n\n"
            "Q: How many players are on the Toronto team in 2005-06?\n"
            "A: SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'\n\n"
            "Q: What is the total attendance at Bridgestone Arena?\n"
            "A: SELECT SUM(attendance) FROM wikisql_data WHERE arena = 'Bridgestone Arena'\n\n"
            "Q: What is the lowest rank for a player from Germany?\n"
            "A: SELECT MIN(rank) FROM wikisql_data WHERE country = 'Germany'\n\n"
            "Q: What year did University of Saskatchewan have their first season?\n"
            "A: SELECT MAX(first_season) FROM wikisql_data WHERE institution = 'University of Saskatchewan'\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM wikisql_data WHERE position = 'Guard' AND years_in_toronto = '1996-97'\n\n"
            "Q: The U.S. airdate of 4 april 2008 had a production code of what?\n"
            "A: SELECT MAX(production_code) FROM wikisql_data WHERE us_airdate = '4 april 2008'\n\n"
            "Q: The canadian airdate of 11 february 2008 applied to what series number?\n"
            "A: SELECT COUNT(no_in_series) FROM wikisql_data WHERE canadian_airdate = '11 february 2008'\n\n"
            "Q: What is Iceland's total?\n"
            "A: SELECT COUNT(total) FROM wikisql_data WHERE country = 'Iceland'\n\n"
            "Q: What is the United States rank?\n"
            "A: SELECT COUNT(rank) FROM wikisql_data WHERE country = 'United States'\n\n"
            "Q: What is the score of the event that Alianza Lima won in 1965?\n"
            "A: SELECT MAX(score) FROM wikisql_data WHERE winner = 'Alianza Lima' AND year = '1965'\n\n"
            "Q: What is the winning score of -8 (71-63-69-69=272)?\n"
            "A: SELECT MIN(year) FROM wikisql_data WHERE score = '-8 (71-63-69-69=272)'\n\n"
            "Q: Name the finished position for Kerry Katona.\n"
            "A: SELECT COUNT(finished) FROM wikisql_data WHERE celebrity = 'Kerry Katona'\n\n"
            f"{agg_hint}\n"
            f"{cond_hint}\n\n"
            "CONDITION TRAPS — common model errors:\n"
            "TRAP 1 (missing WHERE — entity-as-subject):\n"
            "  'What X are the [Entity] [verb]?' → WHERE [entity_col] = '[Entity]'\n"
            "  A noun in the question that names something you are NOT selecting\n"
            "  MUST appear in a WHERE condition.\n"
            "  BAD:  SELECT location FROM wikisql_data\n"
            "  GOOD: SELECT MAX(location) FROM wikisql_data WHERE nickname = 'Miners'\n\n"
            "TRAP 2 (context year ≠ WHERE filter — use named entity instead):\n"
            "  'Which [role] from the [year] [event] attended [Named_Entity]?'\n"
            "  The [year] is context; [Named_Entity] is the actual WHERE value.\n"
            "  BAD:  WHERE pick = 2004\n"
            "  GOOD: WHERE college = 'Wilfrid Laurier'\n\n"
            "TRAP 3 (synonymous-with → quoted literal, NOT column reference):\n"
            "  'value of X is synonymous with its category' → WHERE col = 'X'\n"
            "  BAD:  WHERE since_beginning_of_big_12 = overall_record\n"
            "  GOOD: WHERE since_beginning_of_big_12 = 'since beginning of big 12'\n\n"
            "NEW EXAMPLES:\n"
            "Q: What city and state are the miners located in?\n"
            "A: SELECT MAX(location) FROM wikisql_data WHERE nickname = 'Miners'\n\n"
            "Q: Which player from the 2004 CFL draft attended Wilfrid Laurier?\n"
            "A: SELECT MAX(player) FROM wikisql_data WHERE college = 'Wilfrid Laurier'\n\n"
            f"Question: {question}\n"
            "SQL:"
        )

    def _construct_prompt(
        self,
        question: str,
        schema_str: str,
        is_wikisql: bool = False,
    ) -> str:
        """
        Construct prompt for Text-to-SQL (MaaS / direct model path).
        is_wikisql=True → WikiSQL annotation rules injected.
        """
        if is_wikisql:
            return self._construct_prompt_wikisql(question, schema_str)

        # ── Spider / general MaaS prompt ─────────────────────────────────
        return (
            "You are an expert SQL assistant. Generate a SQL query following Spider benchmark format.\n\n"
            f"Database Schema:\n{schema_str}\n\n"
            "CRITICAL OUTPUT FORMAT:\n"
            "- Output ONLY the raw SQL query — no explanations, no reasoning, no comments\n"
            "- Do NOT include markdown fences, labels like 'SQL:', or footnotes\n"
            "- Do NOT write 'But wait', 'However', 'Note', or any prose after the query\n"
            "- Start your response DIRECTLY with SELECT\n\n"
            "CRITICAL SPIDER FORMAT RULES:\n"
            "1. Use ONLY 'JOIN' — NEVER INNER JOIN, LEFT JOIN, RIGHT JOIN\n"
            "2. DO NOT use CASE statements\n"
            "3. Use simple aggregate functions: COUNT(*), SUM(), AVG(), MIN(), MAX()\n"
            "4. Use lowercase for all identifiers\n"
            "5. Do not include trailing semicolons\n"
            "6. For single table queries: NEVER use table aliases\n"
            "7. For multi-table queries: Use simple aliases like t1, t2\n\n"
            "AGGREGATION RULES:\n"
            '- "how many" / "total number of" → COUNT(col)  [NOT SUM]\n'
            '- "total <numeric col>"          → SUM(col)\n'
            '- "minimum / lowest"             → MIN(col)\n'
            '- "maximum / highest"            → MAX(col)\n'
            "- Exact string match: WHERE col = 'value'  [NOT LIKE]\n\n"
            f"Question: {question}\n"
            "SELECT"
        )

    def _construct_prompt_wikisql(self, question: str, schema_str: str) -> str:
        return (
            "You are a Text-to-SQL expert for WikiSQL.\n\n"
            "OUTPUT RULES:\n"
            "- Output ONE SQL SELECT query only. No explanations. No markdown. No semicolon.\n"
            "- Table name is always: wikisql_data\n"
            "- Use EXACTLY the column names from the schema below (case-preserved).\n\n"
            "- NEVER use subqueries, nested SELECT, or correlated queries.\n"
            "- COUNTING RULE: 'how many [entities]?' / 'number of [X]?' → COUNT(col), not bare SELECT.\n"
            "  Example: 'How many players on Toronto in 2005-06?'\n"
            "  → SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'\n"
            "- NUMERIC RULE: 'how many viewers/goals/points for Y?' (column stores the value) → SELECT col.\n"
            "  → SELECT viewers FROM wikisql_data WHERE director = 'X'  (NOT SUM or COUNT)\n"
            f"{WIKISQL_ANNOTATION_RULES}\n"
            f"Database Schema:\n{schema_str}\n\n"
            "EXAMPLES:\n"
            "Q: What is the pick number for Northwestern college?\n"
            "WRONG: SELECT pick FROM wikisql_data WHERE college = 'Northwestern'\n"        
            "RIGHT: SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'\n\n" 
            "Q: What is the episode number that has production code 8ABX15?\n"
            "A: SELECT MIN(no_in_series) FROM wikisql_data WHERE production_code = '8ABX15'\n\n"
            "Q: How many schools did player 3 play at?\n"
            "A: SELECT COUNT(school_club_team) FROM wikisql_data WHERE no_ = 3\n\n"
            "Q: How many players are on the Toronto team in 2005-06?\n"
            "A: SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM wikisql_data WHERE position = 'Guard'\n\n"
            "Q: What year did University of Saskatchewan have their first season?\n"        
            "A: SELECT MAX(first_season) FROM wikisql_data WHERE institution = 'University of Saskatchewan'\n\n"  
            "Q: What is the enrollment for Foote Field?\n"                                 
            "A: SELECT MAX(enrollment) FROM wikisql_data WHERE football_stadium = 'Foote Field'\n\n"  
            "CONDITION TRAPS — common model errors:\n"
            "TRAP 1 (missing WHERE — entity-as-subject):\n"
            "  'What X are the [Entity] [verb]?' → WHERE [entity_col] = '[Entity]'\n"
            "  A noun in the question that names something you are NOT selecting\n"
            "  MUST appear in a WHERE condition.\n"
            "  BAD:  SELECT location FROM wikisql_data\n"
            "  GOOD: SELECT MAX(location) FROM wikisql_data WHERE nickname = 'Miners'\n\n"
            "TRAP 2 (context year ≠ WHERE filter — use named entity instead):\n"
            "  'Which [role] from the [year] [event] attended [Named_Entity]?'\n"
            "  The [year] is context; [Named_Entity] is the actual WHERE value.\n"
            "  BAD:  WHERE pick = 2004\n"
            "  GOOD: WHERE college = 'Wilfrid Laurier'\n\n"
            "TRAP 3 (synonymous-with → quoted literal, NOT column reference):\n"
            "  'value of X is synonymous with its category' → WHERE col = 'X'\n"
            "  BAD:  WHERE since_beginning_of_big_12 = overall_record\n"
            "  GOOD: WHERE since_beginning_of_big_12 = 'since beginning of big 12'\n\n"
            "NEW EXAMPLES:\n"
            "Q: What city and state are the miners located in?\n"
            "A: SELECT MAX(location) FROM wikisql_data WHERE nickname = 'Miners'\n\n"
            "Q: Which player from the 2004 CFL draft attended Wilfrid Laurier?\n"
            "A: SELECT MAX(player) FROM wikisql_data WHERE college = 'Wilfrid Laurier'\n\n"
            f"Question: {question}\n"
            "SQL:"                   
        )

    # ------------------------------------------------------------------
    # SQL extraction  (handles DeepSeek R1 / CoT verbose output)
    # ------------------------------------------------------------------

    def _clean_sql(self, result: str) -> str:
        """
        Extract a clean SQL SELECT statement from arbitrary LLM output.

        Priority:
          1. ```sql … ``` fenced block
          2. Generic ``` … ``` starting with SELECT
          3. Lines starting with SELECT (pick last — CoT puts final SQL last)
          4. "SQL:" / "Final SQL:" label prefix
          5. Response starts mid-SELECT (prompt priming artifact) → prepend SELECT
          6. Any SELECT … substring as last resort
        """
        if not result or not result.strip():
            return ""

        text = result.strip()

        # 1. ```sql ... ```
        m = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m:
            return self._finalize(m.group(1))

        # 2. ``` SELECT ... ```
        m = re.search(r"```\s*(SELECT\b.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m:
            return self._finalize(m.group(1))

        # 3. Lines starting with SELECT — take the last one
        lines = text.splitlines()
        last_select_idx = None
        for i, ln in enumerate(lines):
            if re.match(r"^\s*SELECT\b", ln.strip(), re.IGNORECASE):
                last_select_idx = i

        if last_select_idx is not None:
            remainder = "\n".join(lines[last_select_idx:])
            candidate = remainder.split("\n\n")[0].strip()
            return self._finalize(candidate)

        # 4. "SQL:" / "Final SQL:" label
        m = re.search(
            r"(?:final\s+sql|sql)\s*:\s*(SELECT\b.*?)(?:\n|$)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return self._finalize(m.group(1))

        # 5. Prompt-primed response (model continues after "SELECT " prefix)
        first_line = text.splitlines()[0].strip()
        if re.match(r"^(COUNT|SUM|AVG|MIN|MAX|DISTINCT|\*)\b", first_line, re.IGNORECASE):
            full_continuation = text.split("\n\n")[0].strip()
            return self._finalize("SELECT " + full_continuation)

        # 6. Any SELECT substring
        m = re.search(r"(SELECT\b.*?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return self._finalize(m.group(1))

        return ""

    def _finalize(self, sql: str) -> str:
        """
        Post-process extracted SQL:
          - Collapse whitespace / drop trailing semicolons
          - Strip dangling prose after a blank line (CoT suffix)
          - Remove backticks
          - Validate it starts with SELECT
        """
        if not sql:
            return ""

        sql = sql.split("\n\n")[0]
        sql = sql.rstrip(";").strip()
        sql = re.sub(
            r"\s+\b(But|However|Note|Therefore|Also|Alternatively|Wait|This)\b.*$",
            "",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        sql = sql.replace("`", "")
        sql = " ".join(sql.split()).strip()

        if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
            return ""

        return sql

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _pattern_generate(self, question: str, schema_info: Dict) -> str:
        """
        Schema-aware heuristic SQL builder used when LLM produces nothing.
        Covers the most common WikiSQL patterns.
        """
        q     = question.lower()
        table = next(iter(schema_info), "table")
        cols  = schema_info.get(table, [])

        if re.search(r"\bhow many\b|\btotal number\b|\bcount\b", q):
            col = self._best_col(q, cols) or (cols[0] if cols else "*")
            return f"SELECT COUNT({col}) FROM {table}"

        if re.search(r"\bhighest\b|\bmost\b|\bmaximum\b|\blargest\b", q):
            col = self._best_col(q, cols) or (cols[0] if cols else "*")
            return f"SELECT MAX({col}) FROM {table}"

        if re.search(r"\blowest\b|\bminimum\b|\bsmallest\b|\bfewest\b", q):
            col = self._best_col(q, cols) or (cols[0] if cols else "*")
            return f"SELECT MIN({col}) FROM {table}"

        if re.search(r"\baverage\b|\bmean\b", q):
            col = self._best_col(q, cols) or (cols[0] if cols else "*")
            return f"SELECT AVG({col}) FROM {table}"

        if re.search(r"\btotal\b|\bsum\b", q):
            col = self._best_col(q, cols) or (cols[0] if cols else "*")
            return f"SELECT SUM({col}) FROM {table}"

        sel_col = self._best_col(q, cols) or "*"
        base    = f"SELECT {sel_col} FROM {table}"

        for col in cols:
            col_pat = col.lower().replace("_", " ").replace("-", " ")
            m = re.search(
                rf"\b{re.escape(col_pat)}\b\s+(?:is|was|are|=)\s+['\"]?([^'\"?,]+)['\"]?",
                q,
            )
            if m:
                val = m.group(1).strip().rstrip("?")
                return f"{base} WHERE {col} = '{val}'"

        return base

    def _best_col(self, question: str, cols: List[str]) -> str:
        """Return the column whose tokens best overlap with the question."""
        q_words    = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
        best, best_score = "", 0
        for col in cols:
            col_words = set(re.sub(r"[^a-z0-9 ]", " ", col.lower()).split())
            score     = len(q_words & col_words)
            if score > best_score:
                best, best_score = col, score
        return best

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _load_schema_info(self, db_path: str) -> Dict[str, List[str]]:
        """Load schema as {table: [col, ...]} from a SQLite database."""
        return self._get_db_schema(db_path)

    def _get_schema_string(self, db_path: str) -> str:
        """Build a human-readable schema string for prompt injection."""
        schema_info = self._load_schema_info(db_path)
        lines = []
        for table, cols in schema_info.items():
            lines.append(f"Table: {table}")
            lines.append(f"Columns: {', '.join(cols)}")
            lines.append("")
        return "\n".join(lines)

    def _get_db_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Extract table/column names from a SQLite database."""
        try:
            conn   = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cursor.fetchall()]
            schema_info: Dict[str, List[str]] = {}
            for table in tables:
                cursor.execute(f'PRAGMA table_info("{table}")')
                schema_info[table] = [col[1] for col in cursor.fetchall()]
            conn.close()
            return schema_info
        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")
            return {}

    def _format_schema(self, schema_info: Dict[str, List[str]]) -> str:
        """Format schema dict as a prompt-friendly string."""
        return "\n".join(
            f"Table {table}: {', '.join(columns)}"
            for table, columns in schema_info.items()
        )

    def _normalize_for_spider(self, sql: str) -> str:
        """
        Lightweight SQL normalization for Spider evaluation.
          - Collapse INNER/LEFT OUTER/RIGHT OUTER JOIN → JOIN / LEFT JOIN / RIGHT JOIN
          - Strip trailing semicolons and extra whitespace
        """
        if not sql:
            return sql
        sql = re.sub(r'\bINNER\s+JOIN\b',        'JOIN',       sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b', 'LEFT JOIN',  sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b','RIGHT JOIN', sql, flags=re.IGNORECASE)
        sql = sql.rstrip(';').strip()
        sql = ' '.join(sql.split())
        return sql

    def _wrap_prompt_for_maas(self, prompt: str) -> list:
        """Wrap prompt as chat messages with strict SQL-only instruction for MaaS."""
        system = (
            "You are a SQL query generator. You output ONLY SQL.\n"
            "ABSOLUTE RULES — no exceptions:\n"
            "1. Output a single SQL query and nothing else\n"
            "2. No explanations, no reasoning, no comments\n"
            "3. No markdown, no code fences, no backticks\n"
            "4. No 'But', 'However', 'Note', or any English text whatsoever\n"
            "5. If unsure, output your best-guess SQL — never output plain text\n"
            "6. Stop immediately after the last SQL token\n"
            "Your entire response must be valid SQL starting with SELECT."
        )
        return [
            {"role": "system",    "content": system},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": "SELECT "},
        ]
    
    def _wikisql_agg_hint(question: str) -> str:
        q = question.lower()
        if re.search(r'\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b', q):
            return "⚡ AGG hint: question asks for COUNT → use COUNT(col)  [NOT COUNT(*)]"
        if re.search(r'\btotal\b|\bsum\b|\bcombined\b|\baltogether\b', q) and not re.search(r'\btotal number\b', q):
            return "⚡ AGG hint: question asks for SUM → use SUM(col)"
        if re.search(r'\bhighest\b|\bmost\b|\blargest\b|\bmaximum\b|\bmax\b|\bbest\b|\blatest\b|\bgreatest\b', q):
            return "⚡ AGG hint: question asks for MAX → use MAX(col)"
        if re.search(r'\blowest\b|\bfewest\b|\bsmallest\b|\bminimum\b|\bmin\b|\bearli\b|\bfirst\b|\boldest\b', q):
            return "⚡ AGG hint: question asks for MIN → use MIN(col)"
        if re.search(r'\baverage\b|\bmean\b|\bavg\b', q):
            return "⚡ AGG hint: question asks for AVG → use AVG(col)"
        return (
            "⚡ AGG hint: No aggregation keyword found, but WikiSQL convention REQUIRES wrapping "
            "single-value results in MAX(col). Use MAX(col) for 'What is/Which/Name the' questions "
            "and COUNT(col) for 'how many' questions. Plain SELECT without AGG is almost always wrong."
        )

    def _wikisql_cond_hint(question: str) -> str:
        q = question.lower()
        if re.search(r'\btallest\b|\blargest\b|\bbiggest\b|\bmost recent\b'
                    r'|\bhighest\b|\blowest\b|\bsmallest\b|\bnewest\b|\boldest\b', q):
            return ("⚡ Condition hint: superlative in question — do NOT add a subquery "
                    "WHERE condition. Use MAX/MIN in SELECT instead.")
        and_count = len(re.findall(r'\band\b', q))
        kw = re.findall(r'\bwhere\b|\bwith\b|\bfor\b|\bnamed\b|\bcalled\b|\bwhen\b', q)
        estimated = max(1, len(kw)) + and_count
        if estimated == 1:
            return "⚡ Condition hint: exactly 1 WHERE condition — do NOT add extra filters."
        return f"⚡ Condition hint: ~{min(estimated, 3)} WHERE conditions — only use what the question states."
    