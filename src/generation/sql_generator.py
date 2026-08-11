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
1. DEFAULT — bare SELECT, no aggregation:
   Most "What is X?" / "Which X?" / "Name the X" questions do NOT need
   MAX/MIN/SUM/COUNT — they just select the matching column.
   → SELECT col FROM wikisql_data WHERE ...
   Example: "What player played guard for Toronto in 1996-97?"
            → SELECT player FROM wikisql_data WHERE position = 'Guard'

2. MAXIMUM — ONLY when the question has an explicit superlative word
   ("highest", "most", "best", "latest", "top", "greatest", "maximum"):
   → SELECT MAX(col) FROM wikisql_data WHERE ...
   Example: "What is the highest pick number for Northwestern?"
            → SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'
   Do NOT use MAX() just because the question starts with "What is the X" —
   only when a superlative word like above is present.

3. MINIMUM — ONLY when the question has an explicit superlative word
   ("lowest", "earliest", "first", "least", "smallest", "minimum"):
   → SELECT MIN(col) FROM wikisql_data WHERE ...

4. COUNTING RECORDS — use COUNT(col), NEVER COUNT(*):
   "How many [entities]?" → SELECT COUNT(col) FROM wikisql_data WHERE ...

5. TOTAL/SUM OVER MULTIPLE ROWS — use SUM() only when the question asks
   for a combined/total value across multiple matching rows
   ("total", "combined", "sum of").

6. WHERE: include ALL filters stated, nothing more. No subqueries. No ORDER BY LIMIT 1.

7. COMPOUND WHERE VALUES — never split on commas:
   WHERE regular_season = '4th, Atlantic Division'  ← correct

8. String values: single quotes. Numeric values: no quotes.

DECISION ORDER: check rules 2/3/4/5 for explicit trigger words first.
If none apply, default to rule 1 (bare SELECT) — this is the MOST COMMON
case. Do not add MAX/MIN/SUM/COUNT unless a trigger word is present.
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
    # Prompt-injection helpers (Semantic Layer + Semantic RAG)
    # ──────────────────────────────────────────────────────────────────────────

    def _format_semantic_hints_block(self, semantic_hints: Optional[str]) -> str:
        """Pass through pre-formatted Semantic Layer hints (see SemanticPipeline)."""
        if not semantic_hints:
            return ""
        return semantic_hints + "\n"

    def _format_few_shot_block(self, few_shot_examples: Optional[List[Dict]]) -> str:
        """Format retrieved few-shot examples (Ek(q)) as a prompt block."""
        if not few_shot_examples:
            return ""
        lines = ["RETRIEVED SIMILAR EXAMPLES (from training set):"]
        for i, ex in enumerate(few_shot_examples, 1):
            q = ex.get('question', ex.get('original_question', ''))
            sql = ex.get('sql_query', ex.get('sql', ''))
            if q and sql:
                lines.append(f"Example {i}:")
                lines.append(f"Q: {q}")
                lines.append(f"SQL: {sql}")
        if len(lines) == 1:
            return ""
        lines.append(
            "\nNote: the examples above are for SQL STRUCTURE only. "
            "For WHERE string values, always copy the EXACT capitalisation "
            "from the CURRENT question below, ignoring how examples wrote theirs."
        )
        lines.append("")
        return "\n".join(lines) + "\n"

    def _ground_literal_casing(self, sql: str, db_path: str) -> str:

        literal_re = re.compile(r"=\s*'([^']*)'")
        matches = list(literal_re.finditer(sql))
        if not matches:
            return sql

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]

            text_columns = []
            for t in tables:
                cur.execute(f"PRAGMA table_info('{t}')")
                for col in cur.fetchall():
                    col_name, col_type = col[1], (col[2] or "").upper()
                    if "CHAR" in col_type or "TEXT" in col_type or "CLOB" in col_type:
                        text_columns.append((t, col_name))

            corrected = sql
            for m in matches:
                value = m.group(1)

                found_values = set()
                for table, col in text_columns:
                    try:
                        cur.execute(
                            f"SELECT DISTINCT trim(\"{col}\") FROM \"{table}\" "
                            f"WHERE trim(lower(\"{col}\")) = trim(lower(?)) LIMIT 2",
                            (value,)
                        )
                        for row in cur.fetchall():
                            if row[0] is not None:
                                found_values.add(str(row[0]))
                    except Exception:
                        continue
                    if len(found_values) > 1:
                        break

                if len(found_values) == 1:
                    canonical = found_values.pop().strip()
                    if canonical and canonical != value:
                        corrected = corrected.replace(f"'{value}'", f"'{canonical}'", 1)

            conn.close()
            return corrected
        except Exception as e:
            logger.debug(f"Value grounding skipped: {e}")
            return sql
    def _normalize_count_star(self, sql: str) -> str:
        def repl(m):
            inner = m.group(1).strip()
            if inner.upper().startswith('DISTINCT'):
                return m.group(0)
            return 'COUNT(*)'
        return re.sub(r'COUNT\s*\(\s*([^)]+?)\s*\)', repl, sql, flags=re.IGNORECASE)

    def _qualify_ambiguous_select_columns(self, sql: str, db_path: str) -> str:
    
        m_select = re.search(r'^SELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s', sql, re.IGNORECASE)
        if not m_select:
            return sql
        select_clause = m_select.group(1)

        alias_order = re.findall(r'(\w+)\s+AS\s+(t\d+)', sql, re.IGNORECASE)
        if len(alias_order) < 2:
            return sql

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            alias_cols = {}
            for table, alias in alias_order:
                cur.execute(f"PRAGMA table_info(\"{table}\")")
                alias_cols[alias] = {r[1].lower() for r in cur.fetchall()}
            conn.close()
        except Exception:
            return sql

        parts = [p.strip() for p in select_clause.split(',')]
        new_parts = []
        changed = False
        for part in parts:
            m = re.match(r'^(\w+)$', part)  # cột trần, không hàm, không đã qualify
            if not m:
                new_parts.append(part)
                continue
            col = m.group(1)
            if col.lower() == '*':
                new_parts.append(part)
                continue
            owners = [a for a, cols in alias_cols.items() if col.lower() in cols]
            if len(owners) >= 2:
                new_parts.append(f"{owners[0]}.{col}")
                changed = True
            else:
                new_parts.append(part)

        if not changed:
            return sql

        new_select = ', '.join(new_parts)
        return sql[:m_select.start(1)] + new_select + sql[m_select.end(1):]

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_alias_columns(self, sql: str, db_path: str) -> Optional[str]:
        alias_map = {}
        for m in re.finditer(r'(\w+)\s+AS\s+(t\d+)', sql, re.IGNORECASE):
            alias_map[m.group(2).lower()] = m.group(1)

        if not alias_map:
            return None

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            table_cols = {}
            for table in set(alias_map.values()):
                cur.execute(f"PRAGMA table_info(\"{table}\")")
                table_cols[table] = {r[1].lower() for r in cur.fetchall()}
            conn.close()
        except Exception:
            return None

        for m in re.finditer(r'\b(t\d+)\.(\w+)\b', sql, re.IGNORECASE):
            alias, col = m.group(1).lower(), m.group(2).lower()
            table = alias_map.get(alias)
            if table and col not in table_cols.get(table, set()):
                return (f"Column '{col}' does not exist in table '{table}' "
                        f"(aliased as {alias}). Check which table actually has this column.")
        return None
    def generate(
        self,
        question: str,
        db_path: str,
        schema_info: Optional[Dict] = None,
        few_shot_examples: Optional[List[Dict]] = None,
        semantic_hints: Optional[str] = None,
        strategy_hints: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate SQL.

        Attempt 1: full prompt (Semantic Layer hints + Semantic RAG few-shot),
                   prefill="SELECT "
        Attempt 2: terse prompt,  prefill="SELECT * FROM "  ← forces FROM clause
        Attempt 3: minimal schema, prefill="SELECT * FROM "
        Fallback:  SELECT 1
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return "SELECT 1"

        schema_str = (
            self._format_schema_dict(schema_info)
            if schema_info else self._get_schema_string(db_path)
        )
        is_wikisql = self._is_wikisql(db_path)

        # Attempt 1: full prompt (Semantic Layer hints + Semantic RAG few-shot)
        prompt = self._construct_prompt(
            question, schema_str, is_wikisql=is_wikisql,
            few_shot_examples=few_shot_examples,
            semantic_hints=semantic_hints,
            strategy_hints=strategy_hints,
        )
        try:
            sql = self._clean_sql(self.model.generate(prompt, prefill="SELECT ", temperature=temperature))
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

        sql = self._normalize_for_spider(sql)
        sql = self._qualify_ambiguous_select_columns(sql, db_path)
        sql = self._normalize_count_star(sql)
        sql = self._ground_literal_casing(sql, db_path)
        alias_error = self._validate_alias_columns(sql, db_path)
        if alias_error:
            logger.warning(f"Alias-column mismatch detected: {alias_error}")
            fix_prompt = (
                "The following SQL has a column error. Fix ONLY the wrong column "
                "reference, keep everything else identical.\n\n"
                f"Schema:\n{schema_str}\n\n"
                f"Question: {question}\n\n"
                f"SQL with error: {sql}\n"
                f"Error: {alias_error}\n\n"
                "Corrected SQL:"
            )
            try:
                fixed = self._clean_sql(self.model.generate(fix_prompt, prefill="SELECT "))
                if fixed and not self._validate_alias_columns(fixed, db_path):
                    sql = self._ground_literal_casing(self._normalize_for_spider(fixed), db_path)
            except Exception as e:
                logger.debug(f"Self-correction retry failed: {e}")
        return sql

    def _format_strategy_block(self, strategy_hints: Optional[str]) -> str:
        """Format ReasoningBank strategies as their OWN prompt block —
        never mixed into the 'Question:' field."""
        if not strategy_hints:
            return ""
        return (
            "REASONING STRATEGIES (learned from prior similar queries — "
            "apply only if relevant, do not force):\n"
            + strategy_hints + "\n"
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Prompt builders
    # ──────────────────────────────────────────────────────────────────────────

    def _construct_prompt(
        self,
        question: str,
        schema_str: str,
        is_wikisql: bool = False,
        few_shot_examples: Optional[List[Dict]] = None,
        semantic_hints: Optional[str] = None,
        strategy_hints: Optional[str] = None,
        max_prompt_tokens: int = 6000,
    ) -> str:
        if is_wikisql:
            return self._construct_prompt_wikisql(
                question, schema_str, few_shot_examples, semantic_hints, strategy_hints)
        return self._construct_prompt_spider(
            question, schema_str, few_shot_examples, semantic_hints, strategy_hints,
            max_prompt_tokens=max_prompt_tokens)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate: ~4 chars/token for English + SQL-mixed text."""
        return len(text) // 4 if text else 0

    def _construct_prompt_spider(
        self,
        question: str,
        schema_str: str,
        few_shot_examples: Optional[List[Dict]] = None,
        semantic_hints: Optional[str] = None,
        strategy_hints: Optional[str] = None,
        max_prompt_tokens: int = 6000,
    ) -> str:
        # ── Phần BẮT BUỘC, không bao giờ bị cắt: rules + schema + question ──
        header = (
            "You are an expert SQL assistant. Generate a SQL query following Spider benchmark format.\n\n"
            "CRITICAL OUTPUT FORMAT:\n"
            "- Output ONLY the raw SQL query — no explanations, no reasoning, no comments\n"
            "- Do NOT include markdown fences, labels like 'SQL:', or footnotes\n"
            "- Start your response DIRECTLY with SELECT\n"
            "- ALWAYS output a complete query — SELECT ... FROM ... at minimum\n\n"
            "CRITICAL SPIDER FORMAT RULES:\n"
            "1. Use ONLY 'JOIN' — NEVER INNER/LEFT/RIGHT JOIN\n"
            "2. DO NOT use CASE statements\n"
            "3. Use lowercase for all identifiers, no trailing semicolons\n"
            "4. Single table queries: NEVER use table aliases\n"
            "5. TABLE ALIASES — STRICT: ONLY t1, t2, t3, t4 (ALWAYS with AS).\n"
            "   Spider parser ONLY resolves tN-style aliases — anything else causes parse errors.\n"
            "   GOOD: FROM pets AS t1 JOIN student AS t2 ON t1.petid = t2.stuid\n"
            "6. COLUMN QUALIFICATION: SELECT/GROUP BY/ORDER BY use bare column names "
            "(no tN.) — EXCEPT when the same column name exists in more than one "
            "joined table (e.g. both tables have 'id', 'name', or the JOIN key "
            "itself like 'document_id'). In that case you MUST qualify with tN. "
            "in SELECT/GROUP BY/ORDER BY too, or SQLite will reject the query with "
            "'ambiguous column name'.\n"
            "   BAD:  SELECT document_id FROM documents AS t1 JOIN paragraphs AS t2 "
            "ON t1.document_id = t2.document_id\n"
            "   GOOD: SELECT t1.document_id FROM documents AS t1 JOIN paragraphs AS t2 "
            "ON t1.document_id = t2.document_id\n"
            "   tN. prefixes ONLY in FROM/JOIN/ON/WHERE.\n"
            "7. MIN/MAX ROW: ORDER BY col ASC/DESC LIMIT 1 — NEVER WHERE col=(SELECT MIN...)\n"
            "8. OR vs UNION: WHERE col=v1 OR col=v2 — NEVER split into UNION\n"
            "9. COLUMN ORDER: exact order from the question\n"
            "10. STRING CASE: exact capitalisation from question in WHERE values\n"
            "11. HAVING vs WHERE: filter aggregates with HAVING after GROUP BY\n"
            "12. SET OPERATORS: INTERSECT='both/shared', EXCEPT='but not/excluding', "
            "UNION='either...or' — NEVER replace with self-JOIN\n"
            "13. DISTINCT only when question says 'unique'/'distinct' — NEVER COUNT(DISTINCT col)\n"
            "14. ALIAS-COLUMN CHECK: before writing tN.column, verify that column "
            "literally belongs to the table aliased as tN in THIS query's FROM/JOIN — "
            "never borrow a column name from a different joined table.\n"
            "\nEXAMPLES:\n"
            "Q: Which model has the smallest horsepower?\n"
            "A: SELECT t1.model FROM car_names AS t1 JOIN cars_data AS t2 ON t1.makeid = t2.id "
            "ORDER BY t2.horsepower ASC LIMIT 1\n\n"
            "Q: Find average and max age for each pet type.\n"
            "A: SELECT avg(pet_age), max(pet_age), pettype FROM pets GROUP BY pettype\n\n"
            "Q: How many pets are owned by students older than 20?\n"
            "A: SELECT COUNT(*) FROM has_pet AS t1 JOIN student AS t2 ON t1.stuid = t2.stuid "
            "WHERE t2.age > 20\n\n"
        )
        schema_block = f"Database Schema:\n{schema_str}\n\n"
        tail = f"Question: {question}\n\nSQL:"

        fixed_cost = self._estimate_tokens(header + schema_block + tail)
        budget_left = max_prompt_tokens - fixed_cost

       
        semantic_block = self._format_semantic_hints_block(semantic_hints)
        few_shot_block = self._format_few_shot_block(few_shot_examples)
        strategy_block = self._format_strategy_block(strategy_hints)

        optional_ordered = [
            ("semantic_hints", semantic_block),
            ("few_shot_examples", few_shot_block),
            ("strategy_hints", strategy_block),
        ]

        included = []
        for name, block in optional_ordered:
            if not block:
                continue
            cost = self._estimate_tokens(block)
            if cost <= budget_left:
                included.append(block)
                budget_left -= cost
            else:
                logger.debug(
                    f"Prompt budget exceeded ({max_prompt_tokens} tok cap) — "
                    f"dropping '{name}' block (~{cost} tok, only {budget_left} left)"
                )

        return header + schema_block + "".join(included) + tail
    
    def _construct_prompt_wikisql(
        self,
        question: str,
        schema_str: str,
        few_shot_examples: Optional[List[Dict]] = None,
        semantic_hints: Optional[str] = None,
        strategy_hints: Optional[str] = None,
    ) -> str:
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
            "Q: What is the highest pick number for Northwestern?\n"
            "A: SELECT MAX(pick) FROM wikisql_data WHERE college = 'Northwestern'\n\n"
            "Q: What is the pick number for Northwestern?\n"
            "A: SELECT pick FROM wikisql_data WHERE college = 'Northwestern'\n\n"
            "Q: How many players on Toronto in 2005-06?\n"
            "A: SELECT COUNT(player) FROM wikisql_data WHERE years_in_toronto = '2005-06'\n\n"
            "Q: What player played guard for Toronto in 1996-97?\n"
            "A: SELECT player FROM wikisql_data WHERE position = 'Guard'\n\n"
            f"{self._format_semantic_hints_block(semantic_hints)}"
            f"{self._format_few_shot_block(few_shot_examples)}"
            f"{self._format_strategy_block(strategy_hints)}"
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