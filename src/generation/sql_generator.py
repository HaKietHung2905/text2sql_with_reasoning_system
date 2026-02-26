"""
SQL Generator using Google GenAI
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path

from src.models.google_genai import GoogleGenAI
from utils.sql_schema import load_schema, Schema
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SQLGenerator:
    """Generates SQL queries from natural language questions using Gemini"""

    def __init__(self, model_name: str = "deepseek-ai/deepseek-r1-0528-maas", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

        use_vertex = os.getenv("USE_VERTEX_AI", "").lower() == "true"
        self.model = GoogleGenAI(
            model_name=model_name,
            api_key=api_key,
            use_vertex_ai=use_vertex,
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1")
        )

    def generate(self, question: str, db_path: str, schema_info: Optional[Dict] = None) -> str:
        """
        Generate SQL for a question and database.

        Args:
            question: Natural language question
            db_path: Path to SQLite database
            schema_info: Optional pre-loaded schema info

        Returns:
            Generated SQL query string
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return ""

        schema_str = self._get_schema_string(db_path)
        prompt = self._construct_prompt(question, schema_str)

        try:
            response = self.model.generate(prompt)
            sql = self._clean_sql(response)

            if not sql:
                logger.error(f"Could not extract SQL for: {question}")
                return ""

            sql = self._normalize_for_spider(sql)
            return sql
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # SQL extraction helpers  (handles DeepSeek R1 / CoT model output)
    # ------------------------------------------------------------------

    def _clean_sql(self, result: str) -> str:
        """
        Extract and clean a single SQL SELECT statement from LLM output.

        Handles:
          - Code-fenced SQL  (```sql ... ```)
          - "SQL:" / "Final SQL:" label prefixes
          - DeepSeek / CoT models that emit reasoning prose before/after
          - Multi-line outputs where the real SQL is buried mid-text

        Returns "" on failure — never returns raw prose.
        """
        if not result or not result.strip():
            return ""

        text = result.strip()

        # 1. ```sql … ``` block (highest confidence)
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

        # 4. Scan every line for SELECT — prefer the *last* one
        #    (CoT models write a draft query first, then the final one)
        select_lines = [
            line.strip()
            for line in text.splitlines()
            if re.match(r"SELECT\b", line.strip(), re.IGNORECASE)
        ]
        if select_lines:
            return self._finalize(select_lines[-1])

        # 5. First SELECT block anywhere in the text
        sql = self._first_select(text)
        if sql:
            return self._finalize(sql)

        # 6. Last resort — run _finalize on the full text, then apply the
        #    prose guard: if the result doesn't start with a SQL keyword,
        #    return "" so the caller can fall back to SELECT 1 cleanly.
        candidate = self._finalize(text)
        if candidate and not re.match(r'^\s*SELECT\b', candidate, re.IGNORECASE):
            logger.warning(f"Could not extract SQL from output: {text[:200]!r}")
            return ""  # signal extraction failure → caller uses SELECT 1
        return candidate

    def _first_select(self, text: str) -> str:
        """Pull the first complete SELECT statement from arbitrary text."""
        # Stop at first blank line — prose usually follows
        m = re.search(r"(SELECT\b.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: stop at semicolon
        m = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    def _finalize(self, sql: str) -> str:
        """
        Final normalisation after extraction:
          - Stop at first blank line / semicolon
          - Drop DeepSeek prose footnotes ("But wait: ...", "However: ...")
          - Remove backticks
          - Collapse whitespace
        """
        if not sql:
            return ""

        # Stop at first blank line — prose usually follows after here
        sql = sql.split("\n\n")[0]

        # Remove trailing semicolon
        sql = sql.split(";")[0]

        # Drop inline prose footnotes that DeepSeek R1 appends
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

    # ------------------------------------------------------------------

    def _normalize_for_spider(self, sql: str) -> str:
        """Normalize SQL to Spider format."""
        if not sql:
            return sql

        sql = re.sub(r'\bINNER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b', 'LEFT JOIN', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b', 'RIGHT JOIN', sql, flags=re.IGNORECASE)

        # Remove trailing semicolon (belt-and-suspenders, _finalize already does this)
        sql = sql.rstrip(';').strip()

        # Collapse whitespace
        sql = ' '.join(sql.split())

        return sql

    def _get_schema_string(self, db_path: str) -> str:
        """Build a human-readable schema string for the prompt."""
        try:
            schema_obj = load_schema(db_path)
            schema_lines = []
            for table, cols in schema_obj.schema.items():
                schema_lines.append(f"Table: {table}")
                schema_lines.append(f"Columns: {', '.join(cols)}")
                schema_lines.append("")
            return "\n".join(schema_lines)
        except Exception as e:
            logger.error(f"Error loading schema for prompt: {e}")
            return ""

    def _construct_prompt(self, question: str, schema_str: str) -> str:
        """Construct prompt for Text-to-SQL with Spider format rules."""
        return f"""You are an expert SQL assistant. Generate a SQL query following Spider benchmark format.

Database Schema:
{schema_str}

CRITICAL SPIDER FORMAT RULES (MUST FOLLOW):
1. Use ONLY 'JOIN' - NEVER use INNER JOIN, LEFT JOIN, RIGHT JOIN, or FULL JOIN
2. DO NOT use CASE statements (not supported by Spider parser)
3. Use WHERE with AND/OR instead of HAVING with CASE
4. Use simple aggregate functions: COUNT(*), SUM(), AVG(), MIN(), MAX()
5. Use lowercase for all identifiers (table names, columns, aliases)
6. Do not include trailing semicolons
7. For single table queries: NEVER use table aliases
8. For multi-table queries: Use simple lowercase aliases like t1, t2
9. For "NOT EXISTS" queries: Use NOT IN with subquery

CRITICAL OUTPUT FORMAT:
- Output ONLY the raw SQL query — no explanations, no reasoning, no comments
- Do NOT include markdown fences (```), labels like "SQL:", or footnotes
- Do NOT write "But wait", "However", "Note", or any prose after the query
- Start your response DIRECTLY with SELECT (or the appropriate SQL keyword)
- Example of a correct response: SELECT player FROM wikisql_data WHERE no_ = 42

Question: {question}

SQL:"""