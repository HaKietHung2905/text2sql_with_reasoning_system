"""
SQL generation from natural language using LangChain.
"""

import os
import re
import sqlite3
from typing import Dict, List, Optional
from dotenv import load_dotenv

from utils.logging_utils import get_logger

# Note: build_sql_prompt does not exist in src.generation.sql_generator —
# we always use _build_prompt defined in this class.

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
                model="deepseek-ai/deepseek-r1-0528-maas",
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
            self.generator = (
                ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
            )
            logger.info("SQL generator (LangChain) initialized")
        except Exception as e:
            logger.error(f"Failed to setup LangChain: {e}")
            self.generator = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, question: str, db_path: str) -> str:
        """
        Generate SQL from question.

        Attempt order:
          1. Full structured prompt  → _clean_sql
          2. Simple/direct prompt   → _clean_sql
          3. _pattern_generate heuristic (schema-aware keyword matching)
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return "SELECT 1"

        schema_info = self._get_db_schema(db_path)
        if not schema_info:
            logger.error("Could not extract schema")
            return "SELECT 1"

        schema_text = self._format_schema(schema_info)

        # Attempt 1: full prompt
        sql = self._invoke(question, schema_text, simple=False)
        if sql:
            return sql

        # Attempt 2: simple prompt
        logger.warning(f"Full prompt produced no SQL for: {question!r} — retrying simple")
        sql = self._invoke(question, schema_text, simple=True)
        if sql:
            return sql

        # Attempt 3: heuristic fallback
        logger.warning(f"Both LLM prompts failed — using heuristic for: {question!r}")
        return self._pattern_generate(question, schema_info)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke(self, question: str, schema_text: str, simple: bool) -> str:
        """Invoke the LangChain chain and return clean SQL, or '' on failure."""
        if not self.generator:
            return ""

        # Always use _build_prompt — never trust the shared import since
        # build_sql_prompt does not exist in src.generation.sql_generator.
        prompt_text = self._build_prompt(question, schema_text, simple=simple)

        try:
            result = self.generator.invoke({"prompt": prompt_text})
            return self._clean_sql(result)
        except Exception as e:
            logger.error(f"LangChain invoke failed ({'simple' if simple else 'full'}): {e}")
            return ""

    def _build_prompt(self, question: str, schema_text: str, simple: bool = False) -> str:
        """
        Build a prompt that forces SQL-only output from CoT/DeepSeek models.

        simple=True  → ultra-terse, used on retry
        simple=False → structured with examples and rules
        """
        if simple:
            # Terse prompt — primes model to continue with SELECT directly
            return (
                f"Schema:\n{schema_text}\n\n"
                f"Question: {question}\n\n"
                "Write a single SQL SELECT. No explanation. No reasoning. "
                "Start with SELECT.\n\nSQL:"
            )

        # IMPORTANT: Do NOT use backtick code fences in this f-string — LangChain's
        # ChatPromptTemplate will try to parse {variable} tokens inside them and
        # will raise a KeyError on the examples.  Use plain text examples instead.
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
            "A: SELECT player FROM table_name WHERE position = 'guard' AND years_in_toronto = '1996-97'\n\n"
            f"Question: {question}\n"
            "SQL Query:"
        )

    # ------------------------------------------------------------------
    # SQL extraction  (handles DeepSeek R1 / CoT verbose output)
    # ------------------------------------------------------------------

    def _clean_sql(self, result: str) -> str:
        """
        Extract a clean SQL SELECT statement from arbitrary LLM output.

        The full prompt ends with "SQL: SELECT", so the model response may
        start mid-statement (e.g. "COUNT(x) FROM y WHERE z").  We prepend
        SELECT in that case.

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
            # Join from that SELECT line to the next blank line (prose boundary)
            remainder = "\n".join(lines[last_select_idx:])
            candidate = remainder.split("\n\n")[0].strip()
            return self._finalize(candidate)


        # 4. "SQL:" / "Final SQL:" label
        m = re.search(
            r"(?:final\s+sql|sql)\s*:\s*(SELECT\b.*?)(?:\n|$)",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            return self._finalize(m.group(1))

        # 5. Prompt-primed response: model continued after "SQL: SELECT"
        #    so output starts with a non-SELECT token like COUNT / *  etc.
        first_line = text.splitlines()[0].strip()
        if re.match(r"^(COUNT|SUM|AVG|MIN|MAX|DISTINCT|\*)\b", first_line, re.IGNORECASE):
            # The whole text is the continuation after the primed "SELECT"
            full_continuation = text.split("\n\n")[0].strip()  # stop at blank line (prose boundary)
            return self._finalize("SELECT " + full_continuation)

        # 6. Any SELECT substring
        m = re.search(r"(SELECT\b.*?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return self._finalize(m.group(1))

        return ""

    def _finalize(self, sql: str) -> str:
        """
        Post-process extracted SQL:
          - Collapse whitespace
          - Drop trailing semicolons / newlines
          - Strip dangling prose after a blank line (CoT suffix)
          - Remove backticks
          - Validate it actually starts with SELECT
        """
        if not sql:
            return ""

        # Split off any prose that follows a blank line
        sql = sql.split("\n\n")[0]
        # Remove trailing semicolons
        sql = sql.rstrip(";").strip()
        # Strip trailing reasoning sentences (DeepSeek R1 appends these)
        sql = re.sub(
            r"\s+\b(But|However|Note|Therefore|Also|Alternatively|Wait|This)\b.*$",
            "",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Remove backticks
        sql = sql.replace("`", "")
        # Collapse whitespace
        sql = " ".join(sql.split()).strip()

        if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
            return ""

        return sql

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _pattern_generate(self, question: str, schema_info: Dict) -> str:
        """
        Schema-aware heuristic SQL builder for when the LLM produces no SQL.

        Covers the most common WikiSQL patterns:
          COUNT, MAX, MIN, SUM, AVG, simple SELECT with optional WHERE.
        """
        q = question.lower()
        table = next(iter(schema_info), "table")
        cols = schema_info.get(table, [])

        # Aggregate detection (order matters — check count before total)
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

        # Plain SELECT + optional WHERE equality
        sel_col = self._best_col(q, cols) or "*"
        base = f"SELECT {sel_col} FROM {table}"

        for col in cols:
            col_pat = col.lower().replace("_", " ").replace("-", " ")
            m = re.search(
                rf"\b{re.escape(col_pat)}\b\s+(?:is|was|are|=)\s+['\"]?([^'\"?,]+)['\"]?",
                q
            )
            if m:
                val = m.group(1).strip().rstrip("?")
                return f"{base} WHERE {col} = '{val}'"

        return base

    def _best_col(self, question: str, cols: List[str]) -> str:
        """Return the column whose tokens best overlap with the question."""
        q_words = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
        best, best_score = "", 0
        for col in cols:
            col_words = set(re.sub(r"[^a-z0-9 ]", " ", col.lower()).split())
            score = len(q_words & col_words)
            if score > best_score:
                best, best_score = col, score
        return best

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _get_db_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Extract table/column names from a SQLite database."""
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
        """Format schema dict as a prompt-friendly string."""
        return "\n".join(
            f"Table {table}: {', '.join(columns)}"
            for table, columns in schema_info.items()
        )
def _construct_prompt(self, question: str, schema_str: str) -> str:
        """Construct prompt for Text-to-SQL — forces SQL-only output for CoT models."""
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
            "EXAMPLES:\n"
            "Q: How many schools did player 3 play at?\n"
            "A: SELECT COUNT(school_club_team) FROM wikisql_data WHERE no_ = 3\n\n"
            "Q: What is the total number of positions on the Toronto team in 2006-07?\n"
            "A: SELECT COUNT(position) FROM wikisql_data WHERE years_in_toronto = '2006-07'\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM wikisql_data WHERE position = 'guard' AND years_in_toronto = '1996-97'\n\n"
            f"Question: {question}\n"
            "SELECT"
        )

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
        # Append "SELECT" to prime the model to continue mid-statement,
        # suppressing CoT reasoning preamble entirely.
        # primed_prompt = prompt.rstrip() + "\nSELECT"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "SELECT "},
        ]