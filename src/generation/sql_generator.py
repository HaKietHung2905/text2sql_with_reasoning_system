"""
src/generation/sql_generator.py  — complete fixed file
"""

import os
import re
import sqlite3
from typing import Optional, Dict, List

from src.models.google_genai import GoogleGenAI
from utils.sql_schema import load_schema
from utils.logging_utils import get_logger

logger = get_logger(__name__)

WIKISQL_ANNOTATION_RULES = """\
━━━ WIKISQL ANNOTATION RULES (follow exactly) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SINGLE-VALUE RETRIEVAL — always wrap in MAX():
   "What is the X?" / "Which X?" / "Name the X"
   → SELECT MAX(col) FROM wikisql_data WHERE ...

2. MINIMUM RETRIEVAL — use MIN() when lowest/earliest/first is implied:
   → SELECT MIN(col) FROM wikisql_data WHERE ...

3. COUNTING RECORDS — use COUNT(col), NEVER COUNT(*):
   "How many [entities]?" → SELECT COUNT(col) FROM wikisql_data WHERE ...

4. NUMERIC VALUE IN A COLUMN — use bare SELECT (NOT SUM, NOT COUNT):
   If column name contains the answer unit (goals, viewers, votes), use SELECT col.

5. TOTAL/SUM OVER MULTIPLE ROWS — use SUM() only across many rows.

6. WHERE: include ALL filters stated, nothing more. No subqueries. No ORDER BY LIMIT 1.

7. COMPOUND WHERE VALUES — never split on commas:
   WHERE regular_season = '4th, Atlantic Division'  ← correct

8. String values: single quotes. Numeric values: no quotes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _is_server_error(exc: Exception) -> bool:
    _5xx = ("500","502","503","504","Internal Server Error","Bad Gateway",
            "Service Unavailable","Gateway Timeout")
    _403 = ("403","Forbidden","BILLING_DISABLED","billing to be enabled")
    seen = set()
    node = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if any(c in str(node) for c in _5xx): return True
        if any(c in repr(node) for c in _5xx): return True
        if any(c in str(node) for c in _403): return True
        if any(c in repr(node) for c in _403): return True
        node = node.__cause__ or node.__context__
    return False


class SQLGenerator:
    """Generates SQL queries from natural language questions using Google GenAI"""

    def __init__(
        self,
        model_name: str = (
            os.getenv("MODEL_NAME")
            or os.getenv("GEMINI_MODEL")
            or "meta/llama-4-maverick-17b-128e-instruct-maas"
        ),
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key    = api_key
        use_vertex = os.getenv("USE_VERTEX_AI", "").lower() == "true"
        self.model = GoogleGenAI(
            model_name=model_name,
            api_key=api_key,
            use_vertex_ai=use_vertex,
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
        )

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
        """
        Generate SQL.

        Attempt 1: full prompt,   prefill="SELECT "
        Attempt 2: terse prompt,  prefill="SELECT * FROM "  ← forces FROM clause
        Attempt 3: minimal schema, prefill="SELECT * FROM "
        Fallback:  SELECT 1
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return "SELECT 1"

        schema_str = self._get_schema_string(db_path)
        is_wikisql = self._is_wikisql(db_path)

        # Attempt 1: full prompt
        prompt = self._construct_prompt(question, schema_str, is_wikisql=is_wikisql)
        try:
            sql = self._clean_sql(self.model.generate(prompt, prefill="SELECT "))
        except Exception as e:
            if _is_server_error(e): raise
            logger.error(f"Generation attempt 1 failed: {e}")
            sql = ""

        # Attempt 2: terse prompt + FROM prefill
        if not sql:
            prompt2 = self._build_terse_prompt(question, schema_str)
            try:
                sql = self._clean_sql(
                    self.model.generate(prompt2, prefill="SELECT * FROM "))
                if sql:
                    logger.info(f"Terse+FROM prefill recovered SQL for: {question!r}")
            except Exception as e:
                if _is_server_error(e): raise
                logger.error(f"Generation attempt 2 (terse+FROM) failed: {e}")
                sql = ""

        # Attempt 3: minimal schema + FROM prefill
        if not sql:
            minimal = self._get_minimal_schema_string(db_path)
            prompt3 = (
                "Write ONE complete SQL SELECT query. Output ONLY SQL.\n"
                "MUST include FROM clause with correct table(s).\n"
                "Use ONLY t1, t2, t3 as aliases (never p, s, hp or other names).\n\n"
                f"Schema:\n{minimal}\n\n"
                f"Question: {question}\n\nSQL:"
            )
            try:
                sql = self._clean_sql(
                    self.model.generate(prompt3, prefill="SELECT * FROM "))
                if sql:
                    logger.info(f"Minimal+FROM prefill recovered SQL for: {question!r}")
            except Exception as e:
                if _is_server_error(e): raise
                logger.error(f"Generation attempt 3 (minimal+FROM) failed: {e}")
                sql = ""

        if not sql:
            logger.error(f"All generation attempts failed for: {question!r} → SELECT 1")
            return "SELECT 1"

        return self._normalize_for_spider(sql)

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt builders
    # ──────────────────────────────────────────────────────────────────────────

    def _construct_prompt(
        self,
        question: str,
        schema_str: str,
        is_wikisql: bool = False,
    ) -> str:
        if is_wikisql:
            return self._construct_prompt_wikisql(question, schema_str)
        return self._construct_prompt_spider(question, schema_str)

    def _construct_prompt_spider(self, question: str, schema_str: str) -> str:
        return (
            "You are an expert SQL assistant. Generate a SQL query following Spider benchmark format.\n\n"
            f"Database Schema:\n{schema_str}\n\n"
            "CRITICAL OUTPUT FORMAT:\n"
            "- Output ONLY the raw SQL query — no explanations, no reasoning, no comments\n"
            "- Do NOT include markdown fences, labels like 'SQL:', or footnotes\n"
            "- Do NOT write 'But wait', 'However', 'Note', or any prose after the query\n"
            "- Start your response DIRECTLY with SELECT\n"
            "- ALWAYS output a complete query — SELECT ... FROM ... at minimum\n\n"
            "CRITICAL SPIDER FORMAT RULES:\n"
            "1. Use ONLY 'JOIN' — NEVER INNER JOIN, LEFT JOIN, RIGHT JOIN\n"
            "2. DO NOT use CASE statements\n"
            "3. Use aggregate functions: COUNT(*), SUM(), AVG(), MIN(), MAX()\n"
            "4. Use lowercase for all identifiers\n"
            "5. No trailing semicolons\n"
            "6. Single table queries: NEVER use table aliases\n"
            # FIX: explicit tN-only rule, ban single-letter aliases
            "7. TABLE ALIASES — STRICT RULES:\n"
            "   ALWAYS define aliases with AS: FROM table AS t1\n"
            "   ONLY use t1, t2, t3, t4 as alias names — NO exceptions.\n"
            "   FORBIDDEN: p, s, a, b, c, hp, cm, ml, cn, cd, T (anything not tN)\n"
            "   Spider parser ONLY resolves tN-style aliases — others cause parse errors\n"
            "   BAD:  FROM pets AS p JOIN student AS s ON p.petid = s.stuid\n"
            "   GOOD: FROM pets AS t1 JOIN student AS t2 ON t1.petid = t2.stuid\n"
            "   BAD:  FROM table t1  (missing AS keyword)\n"
            "   GOOD: FROM table AS t1\n"
            "8. COLUMN QUALIFICATION:\n"
            "   SELECT, GROUP BY, ORDER BY: bare column names only — no tN. prefix\n"
            "   tN. prefixes allowed ONLY in FROM/JOIN/ON/WHERE\n"
            "   BAD:  SELECT t1.name, t2.country GROUP BY t1.name\n"
            "   GOOD: SELECT name, country       GROUP BY name\n"
            "9. MIN/MAX ROW: use ORDER BY col ASC/DESC LIMIT 1\n"
            "   NEVER WHERE col=(SELECT MIN(col)...) — returns duplicates\n"
            "10. OR vs UNION: WHERE col=v1 OR col=v2 — NEVER split into UNION\n"
            "11. COLUMN ORDER: exact order from the question\n"
            "    Q: 'average and max age for each type' → SELECT avg(age), max(age), pettype\n"
            "12. STRING CASE: exact capitalisation from question in WHERE values\n"
            "    Q: 'singers from France' → WHERE country = 'France'  NOT 'france'\n"
            "13. COLUMN vs FUNCTION: if schema has column 'average', use it — don't replace with avg()\n"
            "14. HAVING vs WHERE: filter aggregates with HAVING after GROUP BY\n"
            "15. SET OPERATORS:\n"
            "    INTERSECT: 'both', 'shared by', 'in both'\n"
            "    EXCEPT:    'but not', 'not in', 'excluding'\n"
            "    UNION:     'either...or', 'all X and all Y'\n"
            "    NEVER replace with self-JOIN\n"
            "16. DISTINCT: only when question says 'unique', 'different', 'distinct'\n"
            "    NEVER COUNT(DISTINCT col) — always COUNT(*)\n"
            "\nEXAMPLES:\n"
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
            f"Question: {question}\n\nSQL:"
        )

    def _construct_prompt_wikisql(self, question: str, schema_str: str) -> str:
        return (
            "You are a Text-to-SQL expert for WikiSQL.\n\n"
            "OUTPUT RULES:\n"
            "- Output ONE SQL SELECT query only. No explanations. No markdown. No semicolon.\n"
            "- Table name is always: wikisql_data\n"
            "- Use EXACTLY the column names from the schema (case-preserved).\n"
            "- NEVER use subqueries or nested SELECT.\n\n"
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

    def _build_terse_prompt(self, question: str, schema_str: str) -> str:
        return (
            "Write ONE complete SQL SELECT query. Output ONLY SQL.\n"
            "MUST include FROM clause. Use ONLY t1, t2, t3 as aliases (never p, s, hp etc).\n\n"
            f"Schema:\n{schema_str}\n\n"
            f"Question: {question}\n\nSQL:"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SQL extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_sql(self, result: str) -> str:
        if not result or not result.strip():
            return ""
        text = result.strip()

        m = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m: return self._finalize(m.group(1))

        m = re.search(r"```\s*(SELECT\b.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m: return self._finalize(m.group(1))

        for prefix in (r"final\s+sql\s*query\s*:", r"final\s+sql\s*:",
                       r"sql\s+query\s*:", r"sql\s*:", r"answer\s*:", r"query\s*:"):
            m = re.search(prefix, text, re.IGNORECASE)
            if m:
                sql = self._first_select(text[m.end():].strip())
                if sql: return self._finalize(sql)

        select_lines = [ln.strip() for ln in text.splitlines()
                        if re.match(r"SELECT\b", ln.strip(), re.IGNORECASE)]
        if select_lines:
            return self._finalize(select_lines[-1])

        sql = self._first_select(text)
        if sql: return self._finalize(sql)
        return ""

    def _first_select(self, text: str) -> str:
        m = re.search(r"(SELECT\b.+?)(?:\n{2,}|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m: return m.group(1).strip()
        m = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
        if m: return m.group(1).strip()
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
        # Reject truncated SQL without FROM
        if not re.search(r"\bFROM\b", sql, re.IGNORECASE):
            logger.warning(f"Rejected truncated SQL (no FROM): {sql!r}")
            return ""
        return sql

    # ──────────────────────────────────────────────────────────────────────────
    # Schema helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_schema_string(self, db_path: str) -> str:
        try:
            schema_obj = load_schema(db_path)
            lines = []
            for table, cols in schema_obj.schema.items():
                lines.append(f"Table: {table}")
                lines.append(f"Columns: {', '.join(cols)}")
                lines.append("")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            fk_lines = []
            for table in tables:
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                for fk in cursor.fetchall():
                    fk_lines.append(f"  {table}.{fk[3]} → {fk[2]}.{fk[4]}")
            conn.close()
            if fk_lines:
                lines.append("Foreign Keys:")
                lines.extend(fk_lines)
            return "\n".join(lines)
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

    def _normalize_for_spider(self, sql: str) -> str:
        if not sql: return sql
        sql = re.sub(r"\bINNER\s+JOIN\b",         "JOIN",       sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bLEFT\s+OUTER\s+JOIN\b",  "LEFT JOIN",  sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bRIGHT\s+OUTER\s+JOIN\b", "RIGHT JOIN", sql, flags=re.IGNORECASE)
        sql = sql.rstrip(";").strip()
        sql = " ".join(sql.split())
        return sql