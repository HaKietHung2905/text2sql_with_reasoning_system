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
            model="deepseek-ai/deepseek-r1-0528-maas",
            temperature=0.1,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # Enhanced prompt template with Spider format rules
        system_prompt = """You are a SQL expert. Generate SIMPLE, correct SQL queries following Spider benchmark format.

Database Schema: {schema}

CRITICAL SPIDER FORMAT RULES (MUST FOLLOW):
1. Use 'JOIN' instead of 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN'
2. DO NOT use CASE statements (not supported by Spider parser)
3. Use WHERE with AND/OR instead of HAVING with CASE
4. Use simple aggregate functions: COUNT(*), SUM(), AVG(), MIN(), MAX()
5. Use lowercase for all identifiers (table names, columns)
6. Do not include trailing semicolons

ADDITIONAL RULES:
- For single table queries: NEVER use table aliases
- For multi-table queries: ALWAYS use simple table aliases (t1, t2, etc.)
- Use COUNT(*) for counting rows
- Verify all table and column names exist in schema

CORRECT EXAMPLES:
✓ SELECT t1.fname FROM student AS t1 WHERE t1.age > 20
✓ SELECT COUNT(*) FROM student
✓ SELECT t1.name FROM stadium AS t1 JOIN concert AS t2 ON t1.stadium_id = t2.stadium_id

INCORRECT EXAMPLES (will fail):
✗ SELECT CASE WHEN condition THEN 1 ELSE 0 END
✗ SELECT t1.fname FROM student AS t1 INNER JOIN pets AS t2
✗ SELECT T1.Fname FROM Student AS T1

Question: {question}

SQL Query:"""
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        self.generator = prompt | llm | StrOutputParser()
        
        logger.info("SQL generator initialized with Spider format rules")
        
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
                cursor.execute(f'PRAGMA table_info("{table}")')
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