"""
SQL generation from natural language using LangChain.
"""

import os
import re
import sqlite3
from typing import Dict, List, Optional
from dotenv import load_dotenv

from utils.logging_utils import get_logger

# Import shared prompt builder from generation module
try:
    from src.generation.sql_generator import build_sql_prompt
    _SHARED_PROMPT_AVAILABLE = True
except ImportError:
    _SHARED_PROMPT_AVAILABLE = False

logger = get_logger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available")


class SQLGenerator:
    """Generate SQL from natural language questions (LangChain / Gemini backend)"""

    def __init__(self):
        self.generator = None
        self.generator_simple = None   # second chain for retry
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()

    def _setup_langchain(self):
        """Setup LangChain chains for SQL generation (full + simple fallback)."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key or api_key == "your-api-key-here":
            logger.warning("Google API key not found")
            return

        try:
            llm = ChatGoogleGenerativeAI(
                model="deepseek-ai/deepseek-r1-0528-maas",
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )

            # ── Full prompt (injected at call time via build_sql_prompt) ──
            full_template = "{prompt}"
            self.generator = (
                ChatPromptTemplate.from_template(full_template) | llm | StrOutputParser()
            )

            # ── Simple/direct prompt (retry fallback) ────────────────────
            self.generator_simple = self.generator  # same chain; prompt differs

            logger.info("SQL generator (LangChain) initialized")

        except Exception as e:
            logger.error(f"Failed to setup LangChain: {e}")
            self.generator = None

    def generate(self, question: str, db_path: str) -> str:
        """
        Generate SQL from question.

        Tries the full structured prompt first; if the result is not valid SQL
        (model produced reasoning prose), retries with a shorter direct prompt.
        If both LLM attempts fail, returns "" so the caller can use SELECT 1.
        Never falls back to pattern matching — a wrong pattern SQL is worse
        than an explicit "" that the caller handles cleanly.
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return ""

        schema_info = self._get_db_schema(db_path)
        if not schema_info:
            logger.error("Could not extract schema")
            return ""

        schema_text = self._format_schema(schema_info)

        # ── Attempt 1: full prompt ─────────────────────────────────────
        sql = self._invoke(question, schema_text, simple=False)
        if sql:
            return sql

        # ── Attempt 2: simple/direct prompt ───────────────────────────
        logger.warning(f"Full prompt produced no SQL for: {question!r} — retrying simple")
        sql = self._invoke(question, schema_text, simple=True)
        if sql:
            return sql

        # ── Both LLM attempts failed — return "" for clean fallback ───
        # Do NOT use _pattern_generate: a generic "SELECT COUNT(*) FROM tbl"
        # silently poisons results.  Let the caller decide (SELECT 1).
        logger.warning(f"All generation attempts failed for: {question!r}")
        return ""

    def _invoke(self, question: str, schema_text: str, simple: bool) -> str:
        """Invoke the LangChain chain and return clean SQL, or "" on failure."""
        if not self.generator:
            return ""

        if _SHARED_PROMPT_AVAILABLE:
            prompt_text = build_sql_prompt(question, schema_text, simple=simple)
        else:
            prompt_text = self._fallback_prompt(question, schema_text, simple)

        try:
            result = self.generator.invoke({"prompt": prompt_text})
            return self._clean_sql(result)
        except Exception as e:
            logger.error(f"LangChain invoke failed ({'simple' if simple else 'full'}): {e}")
            return ""

    def _fallback_prompt(self, question: str, schema_text: str,
                         simple: bool = False) -> str:
        """Inline prompt builder used when src.generation is not importable."""
        if simple:
            return f"""Database Schema:
{schema_text}

Write a single SQL SELECT statement for the question below.
Rules:
- Output ONLY the SQL, nothing else. Start with SELECT.
- Use = for exact matches (not LIKE).
- "how many" / "total number of" → COUNT(col)
- "total amount of [countable]"  → COUNT(col), NOT SUM
- "minimum X" → MIN(col), "maximum X" → MAX(col)
- Never split a compound filter like '4th, Atlantic Division' into two conditions.

Question: {question}
SQL:"""

        return f"""You are a SQL expert. Output ONLY a single SQL SELECT statement — nothing else.
Do not explain. Do not reason. Do not add footnotes. Start directly with SELECT.

Database Schema:
{schema_text}

══ HARD RULES ══════════════════════════════════════════════════════════

OUTPUT FORMAT
  • First character MUST be 'S' (SELECT)
  • NO markdown, NO labels, NO comments, NO prose
  • One line, no trailing semicolon

AGGREGATION
  • "how many X" / "total number of X"       → COUNT(x_column)  ← NOT SUM
  • "total amount of X" (X is a count)        → COUNT(x_column)  ← NOT SUM
  • "total amount / sum of X" (X is numeric)  → SUM(x_column)
  • "minimum / lowest / Name the min X"       → MIN(x_column)    ← ALWAYS wrap
  • "maximum / highest / total X when ..."    → MAX(x_column)    ← ALWAYS wrap
  • "what is the X when <filter>" (single)    → MIN(x_column) WHERE <filter>

WHERE CLAUSE
  • Exact match: WHERE col = 'value'   NOT LIKE
  • Dates/years: WHERE years_in_toronto = '1996-97'   NOT LIKE '%1996-97%'
  • Compound value stays as one string:
      WHERE regular_season = '4th, Atlantic Division'  ← CORRECT
      WHERE regular_season = '4th' AND division = ...  ← WRONG

══ EXAMPLES ════════════════════════════════════════════════════════════
Q: How many schools did player 3 play at?
A: SELECT COUNT(school_club_team) FROM wikisql_data WHERE no_ = 3

Q: What is the total amount of numbers on the Toronto team in 2005-06?
A: SELECT COUNT(no_) FROM wikisql_data WHERE years_in_toronto = '2005-06'

Q: Name the minimum ties played for 6 years.
A: SELECT MIN(ties_played) FROM wikisql_data WHERE years_played = 6

Q: What player played guard for Toronto in 1996-97?
A: SELECT player FROM wikisql_data WHERE position = 'guard' AND years_in_toronto = '1996-97'

══════════════════════════════════════════════════════════════════════

Question: {question}
SQL:"""

    # ------------------------------------------------------------------
    # SQL extraction  (handles DeepSeek R1 / CoT verbose output)
    # ------------------------------------------------------------------

    def _clean_sql(self, result: str) -> str:
        """Extract and clean a SQL statement. Returns "" on failure (never prose)."""
        if not result or not result.strip():
            return ""

        text = result.strip()

        # 1. ```sql … ``` block
        m = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m:
            return self._finalize(m.group(1))

        # 2. Generic ``` … ``` that starts with SELECT
        m = re.search(r"```\s*(SELECT\b.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if m:
            return self._finalize(m.group(1))

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
                sql = self._first_select(candidate)
                if sql:
                    return self._finalize(sql)

        # 4. Last SELECT-starting line (prefer last — CoT writes draft first)
        select_lines = [
            line.strip()
            for line in text.splitlines()
            if re.match(r"SELECT\b", line.strip(), re.IGNORECASE)
        ]
        if select_lines:
            return self._finalize(select_lines[-1])

        # 5. First SELECT block anywhere
        sql = self._first_select(text)
        if sql:
            return self._finalize(sql)

        # 6. Nothing found — apply prose guard then return "" if still not SQL
        candidate = self._finalize(text)
        if candidate and not re.match(r'^\s*SELECT\b', candidate, re.IGNORECASE):
            logger.warning(f"Could not extract SQL from output: {text[:200]!r}")
            return ""  # signal extraction failure → caller uses SELECT 1
        return candidate

    def _first_select(self, text: str) -> str:
        """Pull the first complete SELECT statement from arbitrary text."""
        m = re.search(r"(SELECT\b.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    def _finalize(self, sql: str) -> str:
        """Normalise extracted SQL: trim prose, remove backticks, collapse whitespace."""
        if not sql:
            return ""
        sql = sql.split("\n\n")[0]
        sql = sql.split(";")[0]
        sql = re.sub(
            r"\s+\b(But|However|Note|Therefore|Also|Alternatively|Wait)\b[^\"']*$",
            "",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        sql = sql.replace("`", "")
        return " ".join(sql.split()).strip()

    # ------------------------------------------------------------------

    def _get_db_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Extract database schema from SQLite file."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cursor.fetchall()]
            schema_info = {}
            for table in tables:
                cursor.execute(f'PRAGMA table_info("{table}")')
                schema_info[table] = [col[1] for col in cursor.fetchall()]
            conn.close()
            return schema_info
        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")
            return {}

    def _format_schema(self, schema_info: Dict[str, List[str]]) -> str:
        """Format schema dict as prompt-friendly string."""
        return '\n'.join(
            f"Table {table}: {', '.join(columns)}"
            for table, columns in schema_info.items()
        )

    def _pattern_generate(self, question: str, schema_info: Dict) -> str:
        """
        Pattern-based SQL generation — kept for reference but NO LONGER called
        from generate().  Calling this silently produces wrong SQLs that corrupt
        evaluation results.  Use SELECT 1 as a safe explicit failure instead.
        """
        q = question.lower()
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT 1"
        tbl = tables[0]
        if re.search(r'how many|count|total number', q):
            return f"SELECT COUNT(*) FROM {tbl}"
        elif re.search(r'list all|show all', q):
            return f"SELECT * FROM {tbl}"
        elif re.search(r'names?', q):
            cols = schema_info.get(tbl, [])
            name_col = next((c for c in cols if 'name' in c.lower()),
                            cols[0] if cols else '*')
            return f"SELECT {name_col} FROM {tbl}"
        else:
            return f"SELECT * FROM {tbl}"