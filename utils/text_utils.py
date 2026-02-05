"""Text processing utility functions"""
import re
from typing import List, Tuple, Dict, Any

def create_tab_separated_line(field1: str, field2: str) -> str:
    """
    Create tab-separated line from two fields
    
    Args:
        field1: First field
        field2: Second field
        
    Returns:
        Tab-separated string with newline
    """
    return f"{field1}\t{field2}\n"


def parse_tab_separated_line(line: str) -> Tuple[str, str]:
    """
    Parse tab-separated line into two fields
    
    Args:
        line: Tab-separated line
        
    Returns:
        Tuple of (field1, field2)
    """
    parts = line.strip().split('\t')
    if len(parts) != 2:
        raise ValueError(f"Expected 2 tab-separated fields, got {len(parts)}")
    return parts[0], parts[1]


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    return ' '.join(text.split())

def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from LLM response
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted SQL query
    """
    # Try to extract from code blocks
    code_block_pattern = r'```sql\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks
    code_block_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Return cleaned response if no code blocks found
    return clean_text(response)

def format_schema_for_prompt(schema: Dict) -> str:
    """
    Format schema dictionary for prompt inclusion
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Formatted schema string
    """
    lines = []
    
    for table in schema.get('tables', []):
        table_name = table.get('name', '')
        columns = table.get('columns', [])
        
        if columns:
            col_str = ', '.join(columns)
            lines.append(f"Table: {table_name} [{col_str}]")
        else:
            lines.append(f"Table: {table_name}")
    
    return '\n'.join(lines)

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_retrieval_results(results: Dict[str, Any], result_type: str = "question") -> str:
    """
    Format retrieval results as readable text
    
    Args:
        results: Results dictionary
        result_type: Type of results ('question', 'sql', 'schema')
        
    Returns:
        Formatted string
    """
    if "error" in results:
        return f"Error: {results['error']}"
    
    lines = []
    query = results.get('query', '')
    lines.append(f"Query: {query}")
    lines.append("-" * 60)
    
    if result_type == "question":
        items = results.get('results', [])
        lines.append(f"Found {len(items)} similar questions")
        
        for item in items:
            lines.append(f"\n{item['rank']}. Similarity: {item['similarity_score']:.3f}")
            lines.append(f"   Question: {item['question']}")
            lines.append(f"   SQL: {item['sql_query']}")
            lines.append(f"   Database: {item['database']}")
    
    elif result_type == "sql":
        items = results.get('results', [])
        lines.append(f"Found {len(items)} similar SQL queries")
        
        for item in items:
            lines.append(f"\n{item['rank']}. Similarity: {item['similarity_score']:.3f}")
            lines.append(f"   SQL: {item['sql_query']}")
            lines.append(f"   Original: {item['original_question']}")
            lines.append(f"   Database: {item['database']}")
    
    elif result_type == "schema":
        schemas = results.get('relevant_schemas', [])
        lines.append(f"Found {len(schemas)} relevant schemas")
        
        for schema in schemas:
            lines.append(f"\n{schema['rank']}. {schema['database']} "
                        f"(Similarity: {schema['similarity_score']:.3f})")
            lines.append(f"   Tables: {schema['tables']}, Columns: {schema['columns']}")
    
    return "\n".join(lines)


def truncate_sql(sql: str, max_length: int = 100) -> str:
    """
    Truncate SQL query for display
    
    Args:
        sql: SQL query
        max_length: Maximum length
        
    Returns:
        Truncated SQL
    """
    if len(sql) <= max_length:
        return sql
    return sql[:max_length - 3] + "..."