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
            use_vertex_ai=use_vertex
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
            return "SELECT 1"
            
        # Load schema
        schema_str = self._get_schema_string(db_path)
        
        # Construct prompt
        prompt = self._construct_prompt(question, schema_str)
        
        # Generate
        try:
            response = self.model.generate(prompt)
            sql = self._clean_sql(response)
            return sql
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "SELECT 1"
    
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
        """Construct prompt for Text-to-SQL"""
        return f"""You are an expert SQL assistant. generate a SQL query to answer the user's question based on the provided database schema.

Database Schema:
{schema_str}

Rules:
1. Return ONLY the SQL query. No markdown, no explanation.
2. Use standard SQLite syntax.
3. If you cannot answer, return SELECT 1.

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
