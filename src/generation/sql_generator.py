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
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize SQL Generator
        
        Args:
            model_name: Gemini model name
            api_key: API key (for Google AI Studio mode)
        """
        self.model_name = model_name
        self.api_key = api_key
        
        # Auto-detect Vertex AI mode from environment
        use_vertex = os.getenv("USE_VERTEX_AI", "").lower() == "true"
        self.model = GoogleGenAI(
            model_name=model_name, 
            api_key=api_key,
            use_vertex_ai=use_vertex,
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1")
        )
        
    def generate(self, question: str, db_path: str, schema_info: Optional[Dict] = None) -> str:
        """
        Generate SQL for a question and database
        
        Args:
            question: Natural language question
            db_path: Path to SQLite database
            schema_info: Optional pre-loaded schema info
            
        Returns:
            Generated SQL query string
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return ""  # Changed from "SELECT 1" to empty string
            
        # Load schema
        schema_str = self._get_schema_string(db_path)
        
        # Construct prompt
        prompt = self._construct_prompt(question, schema_str)
        
        # Generate
        try:
            response = self.model.generate(prompt)
            sql = self._clean_sql(response)
            
            # Validate SQL
            if not sql or sql.strip().upper() == "SELECT 1":
                logger.error(f"Invalid SQL generated for: {question}")
                return ""
            
            # Apply Spider normalization
            sql = self._normalize_for_spider(sql)
            
            return sql
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""  

    def _normalize_for_spider(self, sql: str) -> str:
        """Normalize SQL to Spider format before returning"""
        if not sql:
            return sql
        
        # Remove INNER/LEFT/RIGHT from JOIN
        sql = re.sub(r'\bINNER\s+JOIN\b', 'JOIN', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bLEFT\s+OUTER\s+JOIN\b', 'LEFT JOIN', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bRIGHT\s+OUTER\s+JOIN\b', 'RIGHT JOIN', sql, flags=re.IGNORECASE)
        
        # Remove trailing semicolon
        sql = sql.rstrip(';').strip()
        
        # Normalize whitespace
        sql = ' '.join(sql.split())
        
        return sql

    def _get_schema_string(self, db_path: str) -> str:
        """From utils.sql_schema.Schema, create a string representation"""
        try:
            # We use the existing load_schema utility
            schema_obj = load_schema(db_path)
            
            # Format:
            # Table: table_name
            # Columns: col1, col2, ...
            schema_lines = []
            for table, cols in schema_obj.schema.items():
                schema_lines.append(f"Table: {table}")
                schema_lines.append(f"Columns: {', '.join(cols)}")
                schema_lines.append("")
                
            return "\\n".join(schema_lines)
        except Exception as e:
            logger.error(f"Error loading schema for prompt: {e}")
            return ""

    def _construct_prompt(self, question: str, schema_str: str) -> str:
        """Construct prompt for Text-to-SQL with Spider format rules"""
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
    9. For "NOT EXISTS" queries: Use NOT IN with subquery, NOT LEFT JOIN with IS NULL
    10. For "EXISTS in both" queries: Use INTERSECT, NOT GROUP BY with HAVING COUNT

    CORRECT PATTERNS:
    ✓ NOT IN pattern: SELECT name FROM table1 WHERE id NOT IN (SELECT id FROM table2)
    ✓ INTERSECT pattern: SELECT col FROM t1 WHERE x = 'a' INTERSECT SELECT col FROM t1 WHERE x = 'b'
    ✓ Simple JOIN: SELECT t1.name FROM table1 AS t1 JOIN table2 AS t2 ON t1.id = t2.id

    INCORRECT PATTERNS (will cause parse errors):
    ✗ LEFT JOIN for NOT EXISTS: SELECT t1.name FROM table1 AS t1 LEFT JOIN table2 AS t2 ... WHERE t2.id IS NULL
    ✗ GROUP BY HAVING COUNT for INTERSECT: SELECT ... GROUP BY ... HAVING COUNT(DISTINCT ...) = 2
    ✗ CASE statements: SELECT CASE WHEN condition THEN 1 ELSE 0 END
    ✗ Mixed case: SELECT T1.Name FROM Student AS T1

    EXAMPLES:
    Question: "Find stadiums that have never hosted a concert"
    ✓ CORRECT: SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)
    ✗ WRONG: SELECT t1.name FROM stadium AS t1 LEFT JOIN concert AS t2 ON t1.id = t2.id WHERE t2.id IS NULL

    Question: "Find items in both categories A and B"  
    ✓ CORRECT: SELECT item FROM items WHERE category = 'A' INTERSECT SELECT item FROM items WHERE category = 'B'
    ✗ WRONG: SELECT item FROM items WHERE category IN ('A', 'B') GROUP BY item HAVING COUNT(DISTINCT category) = 2

    Question: {question}

    SQL Query:"""

    def _clean_sql(self, text: str) -> str:
        """Clean generated text to extract SQL and normalize whitespace"""
        # Remove markdown code blocks (sql, sqlite, or just ```)
        text = re.sub(r'```(?:sql|sqlite)?', '', text, flags=re.IGNORECASE)
        text = text.replace('```', '')
        text = text.strip()
        
        # Extract SQL starting from SELECT (handles prefixes like "SQLite " or "ite ")
        match = re.search(r'(SELECT.*)', text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1)
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
