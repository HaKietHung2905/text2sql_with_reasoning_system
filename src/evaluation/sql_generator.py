"""
src/evaluation/sql_generator.py  — complete fixed file (LangChain path)
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
   "What is the X?" / "Which X?" / "Name the X"
   → SELECT MAX(col) FROM wikisql_data WHERE ...
   Examples:
     "What is the pick number for Northwestern?"
       → SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'

2. MINIMUM RETRIEVAL — use MIN() when lowest/earliest/first is implied:
   → SELECT MIN(col) FROM wikisql_data WHERE ...

3. COUNTING RECORDS — use COUNT(col), NEVER COUNT(*):
   "How many [entities]?" → SELECT COUNT(col) FROM wikisql_data WHERE ...
   NEVER COUNT(*), always COUNT(specific_column).

4. NUMERIC VALUE IN A COLUMN — use bare SELECT (NOT SUM, NOT COUNT):
   If column name contains the answer unit (goals, viewers, votes), use SELECT col.

5. TOTAL/SUM OVER MULTIPLE ROWS — use SUM() only across many rows.

6. WHERE: include ALL filters stated, nothing more. No subqueries. No ORDER BY LIMIT 1.

7. COMPOUND WHERE VALUES — never split on commas:
   WHERE regular_season = '4th, Atlantic Division'  ← correct

8. String values: single quotes. Numeric values: no quotes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _wikisql_agg_hint(question: str) -> str:
    q = question.lower()
    if re.search(r'\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b', q):
        return "⚡ AGG hint: COUNT → use COUNT(col)"
    if re.search(r'\btotal\b|\bsum\b', q) and not re.search(r'\btotal number\b', q):
        return "⚡ AGG hint: SUM → use SUM(col)"
    if re.search(r'\bhighest\b|\bmost\b|\blargest\b|\bmaximum\b|\bmax\b|\blatest\b', q):
        return "⚡ AGG hint: MAX → use MAX(col)"
    if re.search(r'\blowest\b|\bfewest\b|\bsmallest\b|\bminimum\b|\bmin\b|\bearli\b|\bfirst\b', q):
        return "⚡ AGG hint: MIN → use MIN(col)"
    if re.search(r'\baverage\b|\bmean\b|\bavg\b', q):
        return "⚡ AGG hint: AVG → use AVG(col)"
    return "⚡ AGG hint: single-value → wrap in MAX(col)"


def _wikisql_cond_hint(question: str) -> str:
    q = question.lower()
    if re.search(r'\btallest\b|\blargest\b|\bbiggest\b|\bhighest\b|\blowest\b|\bsmallest\b', q):
        return "⚡ Condition hint: superlative — use MAX/MIN in SELECT, NOT a WHERE subquery."
    return "⚡ Condition hint: only use WHERE conditions explicitly stated in the question."


class SQLGenerator:
    """Generate SQL from natural language questions (LangChain / Gemini backend)"""

    def __init__(self):
        self.generator = None
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()

    def _setup_langchain(self):
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

    # ──────────────────────────────────────────────────────────────────────────
    # WikiSQL detection
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_wikisql(db_path: str) -> bool:
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

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        db_path: str,
        schema_info: Optional[Dict] = None,
    ) -> str:
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return "SELECT 1"

        if schema_info is None:
            schema_info = self._load_schema_info(db_path)

        schema_str = self._get_schema_string(db_path)
        is_wikisql = self._is_wikisql(db_path)

        # Attempt 1: full prompt
        sql = self._invoke(question, schema_str, simple=False, is_wikisql=is_wikisql)

        # Attempt 2: terse prompt
        if not sql:
            sql = self._invoke(question, schema_str, simple=True, is_wikisql=is_wikisql)
            if sql:
                logger.info(f"Terse prompt recovered SQL for: {question!r}")

        # Attempt 3: minimal schema
        if not sql:
            minimal = self._get_minimal_schema_string(db_path)
            sql = self._invoke(question, minimal, simple=True, is_wikisql=is_wikisql)
            if sql:
                logger.info(f"Minimal schema recovered SQL for: {question!r}")

        # Attempt 4: pattern fallback
        if not sql:
            logger.warning(f"LLM returned no SQL for: {question!r} → pattern fallback")
            try:
                sql = self._pattern_generate(question, schema_info)
            except Exception as e:
                logger.error(f"Pattern fallback failed: {e}")

        if not sql:
            logger.error(f"All generation methods failed for: {question!r} → SELECT 1")
            return "SELECT 1"

        return self._normalize_for_spider(sql)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _invoke(
        self,
        question: str,
        schema_text: str,
        simple: bool = False,
        is_wikisql: bool = False,
    ) -> str:
        if not self.generator:
            return ""
        prompt_text = self._build_prompt(
            question, schema_text, simple=simple, is_wikisql=is_wikisql)
        try:
            result = self.generator.invoke({"prompt": prompt_text})
            return self._clean_sql(result)
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in (
                "500","502","503","504",
                "Internal Server Error","Bad Gateway",
                "Service Unavailable","Gateway Timeout",
            )):
                raise
            logger.error(f"LangChain invoke failed ({'simple' if simple else 'full'}): {e}")
            return ""

    def _build_prompt(
        self,
        question: str,
        schema_text: str,
        simple: bool = False,
        is_wikisql: bool = False,
    ) -> str:
        if simple:
            return (
                f"Schema:\n{schema_text}\n\n"
                f"Question: {question}\n\n"
                "Write a single SQL SELECT. No explanation. No reasoning. "
                "Use ONLY t1, t2, t3 as aliases. "
                "Start with SELECT.\n\nSQL:"
            )

        if is_wikisql:
            return self._build_prompt_wikisql(question, schema_text)

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
            # FIX: ban single-letter aliases explicitly
            "- TABLE ALIASES: always use AS. ONLY t1, t2, t3 etc — NEVER p, s, hp or other single-letter names.\n"
            "    BAD:  FROM pets AS p  /  FROM student AS s  ← Spider parser FAILS on these\n"
            "    GOOD: FROM pets AS t1 / FROM student AS t2\n"
            "    BAD:  FROM table t1  (no AS keyword)\n"
            "    GOOD: FROM table AS t1\n"
            "- COLUMN QUALIFICATION: in SELECT/GROUP BY/ORDER BY use bare column names only.\n"
            "    tN. prefixes allowed ONLY in FROM/JOIN/ON/WHERE.\n"
            "    BAD:  SELECT t1.name, t2.country GROUP BY t1.name\n"
            "    GOOD: SELECT name, country       GROUP BY name\n"
            "- MIN/MAX ROW: ORDER BY col ASC/DESC LIMIT 1 — NEVER WHERE col=(SELECT MIN(col))\n"
            "- OR vs UNION: WHERE col=v1 OR col=v2 — NEVER split into UNION\n"
            "- COLUMN ORDER: exact order from the question\n"
            "- STRING CASE: exact capitalisation from question in WHERE values\n"
            "- COLUMN vs FUNCTION: if schema has column 'average', use it — don't replace with avg()\n"
            "- HAVING: filter aggregates with HAVING after GROUP BY, not WHERE\n"
            "- SET OPERATORS: INTERSECT/EXCEPT/UNION — NEVER replace with self-JOIN\n"
            "- DISTINCT: only when question says 'unique', 'different', 'distinct'\n"
            "- NEVER COUNT(DISTINCT col) — always COUNT(*)\n"
            "- Exact string match: WHERE col = 'value'  [NOT LIKE]\n"
            "- Keep compound values together: WHERE col = '4th, Atlantic Division'\n"
            "- Never invent columns or tables not in the schema.\n\n"
            "EXAMPLES:\n"
            "Q: Which model has the smallest horsepower?\n"
            "A: SELECT t1.model FROM car_names AS t1 JOIN cars_data AS t2 ON t1.makeid = t2.id "
            "ORDER BY t2.horsepower ASC LIMIT 1\n\n"
            "Q: How many concerts in 2014 or 2015?\n"
            "A: SELECT COUNT(*) FROM concert WHERE year = 2014 OR year = 2015\n\n"
            "Q: Find average and max age for each pet type.\n"
            "A: SELECT avg(pet_age), max(pet_age), pettype FROM pets GROUP BY pettype\n\n"
            "Q: What is the maximum capacity and average of all stadiums?\n"
            "A: SELECT max(capacity), average FROM stadium\n\n"
            "Q: How many pets are owned by students older than 20?\n"
            "A: SELECT COUNT(*) FROM has_pet AS t1 JOIN student AS t2 ON t1.stuid = t2.stuid "
            "WHERE t2.age > 20\n\n"
            "Q: Find the weight of the youngest dog.\n"
            "A: SELECT weight FROM pets WHERE pettype = 'dog' ORDER BY pet_age ASC LIMIT 1\n\n"
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
            "- NEVER use subqueries, nested SELECT, or correlated queries.\n\n"
            f"{WIKISQL_ANNOTATION_RULES}\n"
            f"Database Schema:\n{schema_text}\n\n"
            f"{agg_hint}\n"
            f"{cond_hint}\n\n"
            "EXAMPLES:\n"
            "Q: What is the pick number for Northwestern?\n"
            "A: SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'\n\n"
            "Q: How many players on Toronto in 2005-06?\n"
            "A: SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM wikisql_data WHERE position = 'Guard'\n\n"
            "Q: What city are the miners located in?\n"
            "A: SELECT MAX(location) FROM wikisql_data WHERE nickname = 'Miners'\n\n"
            f"Question: {question}\n"
            "SQL:"
        )

    def _construct_prompt(
        self,
        question: str,
        schema_str: str,
        is_wikisql: bool = False,
    ) -> str:
        if is_wikisql:
            return self._construct_prompt_wikisql(question, schema_str)
        return (
            "You are an expert SQL assistant. Generate a SQL query following Spider benchmark format.\n\n"
            f"Database Schema:\n{schema_str}\n\n"
            "CRITICAL OUTPUT FORMAT:\n"
            "- Output ONLY the raw SQL query — no explanations, no reasoning, no comments\n"
            "- Do NOT include markdown fences, labels like 'SQL:', or footnotes\n"
            "- Do NOT write 'But wait', 'However', 'Note', or any prose after the query\n"
            "- Start your response DIRECTLY with SELECT\n"
            "- ALWAYS output a complete query — never stop mid-query\n\n"
            "CRITICAL SPIDER FORMAT RULES:\n"
            "1. Use ONLY 'JOIN' — NEVER INNER JOIN, LEFT JOIN, RIGHT JOIN\n"
            "2. DO NOT use CASE statements\n"
            "3. Use simple aggregate functions: COUNT(*), SUM(), AVG(), MIN(), MAX()\n"
            "4. Use lowercase for all identifiers\n"
            "5. Do not include trailing semicolons\n"
            "6. For single table queries: NEVER use table aliases\n"
            # FIX: explicit tN-only, ban single-letter aliases
            "7. TABLE ALIASES — STRICT RULES:\n"
            "   ALWAYS define aliases with AS: FROM table AS t1\n"
            "   ONLY use t1, t2, t3, t4 as alias names — NO exceptions.\n"
            "   FORBIDDEN: p, s, a, b, c, hp, cm, ml, cn, cd — Spider parser ignores these\n"
            "   BAD:  FROM pets AS p JOIN student AS s  ← parse error\n"
            "   GOOD: FROM pets AS t1 JOIN student AS t2\n"
            "8. COLUMN QUALIFICATION:\n"
            "   SELECT, GROUP BY, ORDER BY: bare column names only — no tN. prefix\n"
            "   tN. prefixes ONLY in FROM/JOIN/ON/WHERE\n"
            "   BAD:  SELECT t1.name, t2.country GROUP BY t1.name\n"
            "   GOOD: SELECT name, country       GROUP BY name\n"
            "9. MIN/MAX ROW: ORDER BY col ASC/DESC LIMIT 1\n"
            "   NEVER WHERE col=(SELECT MIN(col)...) — returns duplicates\n"
            "10. OR vs UNION: WHERE col=v1 OR col=v2 — NEVER split into UNION\n"
            "11. COLUMN ORDER: exact order from the question\n"
            "12. STRING CASE: exact capitalisation from question in WHERE values\n"
            "13. COLUMN vs FUNCTION: if schema has column 'average', use it directly\n"
            "14. HAVING vs WHERE: HAVING for aggregated values after GROUP BY\n"
            "15. SET OPERATORS: INTERSECT/EXCEPT/UNION — NEVER replace with self-JOIN\n"
            "16. DISTINCT: only for 'unique', 'different', 'distinct'. NEVER COUNT(DISTINCT col)\n"
            "\nEXAMPLES:\n"
            "Q: Which model has the smallest horsepower?\n"
            "A: SELECT t1.model FROM car_names AS t1 JOIN cars_data AS t2 ON t1.makeid = t2.id "
            "ORDER BY t2.horsepower ASC LIMIT 1\n\n"
            "Q: How many concerts in 2014 or 2015?\n"
            "A: SELECT COUNT(*) FROM concert WHERE year = 2014 OR year = 2015\n\n"
            "Q: Find average and max age for each pet type.\n"
            "A: SELECT avg(pet_age), max(pet_age), pettype FROM pets GROUP BY pettype\n\n"
            "Q: How many pets are owned by students older than 20?\n"
            "A: SELECT COUNT(*) FROM has_pet AS t1 JOIN student AS t2 ON t1.stuid = t2.stuid "
            "WHERE t2.age > 20\n\n"
            f"Question: {question}\n\n"
            "SQL:"
        )

    def _construct_prompt_wikisql(self, question: str, schema_str: str) -> str:
        return (
            "You are a Text-to-SQL expert for WikiSQL.\n\n"
            "OUTPUT RULES:\n"
            "- Output ONE SQL SELECT query only. No explanations. No markdown. No semicolon.\n"
            "- Table name is always: wikisql_data\n"
            "- Use EXACTLY the column names from the schema below (case-preserved).\n"
            "- NEVER use subqueries, nested SELECT, or correlated queries.\n\n"
            f"{WIKISQL_ANNOTATION_RULES}\n"
            f"Database Schema:\n{schema_str}\n\n"
            "EXAMPLES:\n"
            "Q: What is the pick number for Northwestern?\n"
            "A: SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'\n\n"
            "Q: How many players on Toronto in 2005-06?\n"
            "A: SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM wikisql_data WHERE position = 'Guard'\n\n"
            f"Question: {question}\n"
            "SQL:"
        )

    def _clean_sql(self, result: str) -> str:
        if not result or not result.strip():
            return ""
        text = result.strip()

        m = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m: return self._finalize(m.group(1))

        m = re.search(r"```\s*(SELECT\b.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m: return self._finalize(m.group(1))

        lines = text.splitlines()
        last_select_idx = None
        for i, ln in enumerate(lines):
            if re.match(r"^\s*SELECT\b", ln.strip(), re.IGNORECASE):
                last_select_idx = i
        if last_select_idx is not None:
            remainder = "\n".join(lines[last_select_idx:])
            candidate = remainder.split("\n\n")[0].strip()
            return self._finalize(candidate)

        m = re.search(r"(?:final\s+sql|sql)\s*:\s*(SELECT\b.*?)(?:\n|$)",
                      text, re.IGNORECASE | re.DOTALL)
        if m: return self._finalize(m.group(1))

        first_line = text.splitlines()[0].strip()
        if re.match(r"^(COUNT|SUM|AVG|MIN|MAX|DISTINCT|\*)\b", first_line, re.IGNORECASE):
            return self._finalize("SELECT " + text.split("\n\n")[0].strip())

        m = re.search(r"(SELECT\b.*?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m: return self._finalize(m.group(1))

        return ""

    def _finalize(self, sql: str) -> str:
        if not sql: return ""
        sql = sql.split("\n\n")[0]
        sql = sql.rstrip(";").strip()
        sql = re.sub(
            r"\s+\b(But|However|Note|Therefore|Also|Alternatively|Wait|This)\b.*$",
            "", sql, flags=re.IGNORECASE | re.DOTALL)
        sql = sql.replace("`", "")
        sql = " ".join(sql.split()).strip()
        if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
            return ""
        if not re.search(r"\bFROM\b", sql, re.IGNORECASE):
            logger.warning(f"Rejected truncated SQL (no FROM): {sql!r}")
            return ""
        return sql

    def _pattern_generate(self, question: str, schema_info: Dict) -> str:
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
                rf"\b{re.escape(col_pat)}\b\s+(?:is|was|are|=)\s+['\"]?([^'\"?,]+)['\"]?", q)
            if m:
                val = m.group(1).strip().rstrip("?")
                return f"{base} WHERE {col} = '{val}'"
        return base

    def _best_col(self, question: str, cols: List[str]) -> str:
        q_words    = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
        best, best_score = "", 0
        for col in cols:
            col_words = set(re.sub(r"[^a-z0-9 ]", " ", col.lower()).split())
            score     = len(q_words & col_words)
            if score > best_score:
                best, best_score = col, score
        return best

    def _load_schema_info(self, db_path: str) -> Dict[str, List[str]]:
        return self._get_db_schema(db_path)

    def _get_schema_string(self, db_path: str) -> str:
        try:
            conn   = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            schema_lines = []
            fk_lines     = []
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                cols = [r[1] for r in cursor.fetchall()]
                schema_lines.append(f"Table: {table} | Columns: {', '.join(cols)}")
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                for fk in cursor.fetchall():
                    fk_lines.append(f"  {table}.{fk[3]} → {fk[2]}.{fk[4]}")
            result = "\n".join(schema_lines)
            if fk_lines:
                result += "\n\nForeign Keys:\n" + "\n".join(fk_lines)
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            return ""

    def _get_minimal_schema_string(self, db_path: str) -> str:
        try:
            conn   = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            lines  = []
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                cols = [r[1] for r in cursor.fetchall()]
                lines.append(f"{table}: {', '.join(cols)}")
            conn.close()
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Minimal schema extraction failed: {e}")
            return ""

    def _get_db_schema(self, db_path: str) -> Dict[str, List[str]]:
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
        return "\n".join(
            f"Table {table}: {', '.join(columns)}"
            for table, columns in schema_info.items()
        )

    def _normalize_for_spider(self, sql: str) -> str:
        if not sql: return sql
        sql = re.sub(r'\bINNER\s+JOIN\b',         'JOIN',       sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b',  'LEFT JOIN',  sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b', 'RIGHT JOIN', sql, flags=re.IGNORECASE)
        sql = sql.rstrip(';').strip()
        sql = ' '.join(sql.split())
        return sql

    def _wrap_prompt_for_maas(self, prompt: str) -> list:
        system = (
            "You are a SQL query generator. You output ONLY SQL.\n"
            "ABSOLUTE RULES — no exceptions:\n"
            "1. Output a single SQL query and nothing else\n"
            "2. No explanations, no reasoning, no comments\n"
            "3. No markdown, no code fences, no backticks\n"
            "4. No 'But', 'However', 'Note', or any English text whatsoever\n"
            "5. ONLY use t1, t2, t3 as table aliases — NEVER p, s, hp or other names\n"
            "6. ALWAYS output a complete query: SELECT ... FROM ...\n"
            "7. Stop immediately after the last SQL token\n"
            "Your entire response must be valid SQL starting with SELECT."
        )
        return [
            {"role": "system",    "content": system},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": "SELECT "},
        ]