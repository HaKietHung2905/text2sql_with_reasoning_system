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

# Check for LangChain availability
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available")


class SQLGenerator:
    """Generate SQL from natural language questions"""
    
    def __init__(self):
        self.generator = None
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()
    
    def _setup_langchain(self):
        """Setup LangChain for SQL generation"""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your-api-key-here":
            logger.warning("Google API key not found")
            return
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
            
            # Basic prompt template
            system_prompt = """You are a SQL expert. Generate SIMPLE, correct SQL queries.

Database Schema: {schema}

CRITICAL RULES:
1. NEVER use table aliases
2. Use simplest possible query
3. Column references: use column_name only for single table
4. Column references: use Table.column_name for multiple tables
5. Return ONLY the SQL query

Question: {question}

SQL Query:"""
            
            prompt = ChatPromptTemplate.from_template(system_prompt)
            self.generator = prompt | llm | StrOutputParser()
            
            logger.info("SQL generator initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup LangChain: {e}")
            self.generator = None
    
    def generate(self, question: str, db_path: str) -> str:
        """
        Generate SQL from question
        
        Args:
            question: Natural language question
            db_path: Path to database file
            
        Returns:
            Generated SQL query
        """
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return "SELECT 1"
        
        schema_info = self._get_db_schema(db_path)
        if not schema_info:
            logger.error("Could not extract schema")
            return "SELECT 1"
        
        schema_text = self._format_schema(schema_info)
        
        if self.generator:
            try:
                result = self.generator.invoke({
                    "question": question,
                    "schema": schema_text
                })
                
                cleaned = self._clean_sql(result)
                return cleaned
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
        
        # Fallback to pattern matching
        return self._pattern_generate(question, schema_info)
    
    def _get_db_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Extract database schema"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            schema_info = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                schema_info[table] = columns
            
            conn.close()
            return schema_info
            
        except Exception as e:
            logger.error(f"Schema extraction failed: {e}")
            return {}
    
    def _format_schema(self, schema_info: Dict[str, List[str]]) -> str:
        """Format schema for prompt"""
        lines = []
        for table, columns in schema_info.items():
            lines.append(f"Table {table}: {', '.join(columns)}")
        return '\n'.join(lines)
    
    def _clean_sql(self, result: str) -> str:
        """Clean LLM result"""
        sql = result.strip()
        
        # Remove "SQL:" prefix
        if sql.lower().startswith("sql:"):
            sql = sql[4:].strip()
        
        # Extract from code blocks
        if "```sql" in sql.lower():
            start = sql.lower().find("```sql") + 6
            end = sql.find("```", start)
            if end != -1:
                sql = sql[start:end].strip()
        elif "```" in sql:
            start = sql.find("```") + 3
            end = sql.find("```", start)
            if end != -1:
                sql = sql[start:end].strip()
        
        # Normalize whitespace
        sql = ' '.join(sql.strip().split())
        
        # Remove trailing semicolon
        if sql.endswith(';'):
            sql = sql[:-1]
        
        return sql
    
    def _pattern_generate(self, question: str, schema_info: Dict) -> str:
        """Pattern-based SQL generation"""
        question_lower = question.lower()
        tables = list(schema_info.keys())
        
        if not tables:
            return "SELECT 1"
        
        primary_table = tables[0]
        
        # Pattern matching
        if re.search(r'how many|count', question_lower):
            return f"SELECT COUNT(*) FROM {primary_table}"
        elif re.search(r'list all|show all', question_lower):
            return f"SELECT * FROM {primary_table}"
        elif re.search(r'names?', question_lower):
            columns = schema_info.get(primary_table, [])
            name_col = next((col for col in columns if 'name' in col.lower()), 
                          columns[0] if columns else '*')
            return f"SELECT {name_col} FROM {primary_table}"
        else:
            return f"SELECT * FROM {primary_table}"