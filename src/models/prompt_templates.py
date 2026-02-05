"""
SQL generation prompt templates.
Provides various prompt strategies for Text-to-SQL conversion.
"""

from enum import Enum
from typing import Dict, List, Optional


class PromptType(Enum):
    """Available prompt template types"""
    BASIC = "basic"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RULE_BASED = "rule_based"
    ENHANCED = "enhanced"
    STEP_BY_STEP = "step_by_step"


class SQLPatterns:
    """Common SQL patterns and their templates"""
    
    PATTERNS = {
        "How many": "SELECT COUNT(*) FROM table",
        "List all": "SELECT * FROM table",
        "Show/Display": "SELECT columns FROM table",
        "Find/Get": "SELECT columns FROM table WHERE condition",
        "Top N/Highest N": "SELECT columns FROM table ORDER BY column DESC LIMIT N",
        "Bottom N/Lowest N": "SELECT columns FROM table ORDER BY column ASC LIMIT N",
        "Average/Mean": "SELECT AVG(column) FROM table",
        "Total/Sum": "SELECT SUM(column) FROM table",
        "Maximum/Highest": "SELECT MAX(column) FROM table",
        "Minimum/Lowest": "SELECT MIN(column) FROM table",
        "Count by group": "SELECT group_column, COUNT(*) FROM table GROUP BY group_column",
        "Contains/Includes": "WHERE column LIKE '%value%'",
        "Starts with": "WHERE column LIKE 'value%'",
        "Ends with": "WHERE column LIKE '%value'",
        "Greater than": "WHERE column > value",
        "Less than": "WHERE column < value",
        "Equal to": "WHERE column = value",
        "Not equal": "WHERE column != value OR column <> value",
        "Between": "WHERE column BETWEEN value1 AND value2"
    }
    
    @classmethod
    def get_patterns_text(cls) -> str:
        """Get formatted patterns text"""
        return "\n".join([
            f"- **\"{pattern}\"** → `{sql}`" 
            for pattern, sql in cls.PATTERNS.items()
        ])


class SQLValidationRules:
    """SQL validation and generation rules"""
    
    RULES = [
        "**Schema Compliance**: Only use tables and columns that exist in the provided schema",
        "**Syntax Correctness**: Generate valid SQL syntax that can be executed without errors",
        "**JOIN Requirements**: Use proper JOIN conditions when accessing multiple tables",
        "**Aggregate Rules**: Include GROUP BY when mixing aggregate and non-aggregate columns",
        "**Data Type Matching**: Use appropriate operators for different data types",
        "**String Quoting**: Properly quote string literals in WHERE conditions",
        "**NULL Handling**: Use IS NULL/IS NOT NULL for null value comparisons",
        "**Column Qualification**: Use table aliases to avoid ambiguous column references in JOINs",
        "**Function Usage**: Use database-appropriate functions (e.g., YEAR(), DATE(), etc.)",
        "**Performance Awareness**: Prefer efficient query structures when multiple approaches are possible",
        "**Single Table Rule**: If query uses single table, do NOT use table alias, column alias, or field alias"
    ]
    
    CRITICAL_RULES = [
        "For single table queries: NO table aliases, NO column aliases, NO field aliases",
        "Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1",
        "Wrong: SELECT t.column1, COUNT(*) as cnt FROM table t GROUP BY t.column1"
    ]
    
    @classmethod
    def get_rules_text(cls) -> str:
        """Get formatted rules text"""
        return "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(cls.RULES)])
    
    @classmethod
    def get_critical_rules_text(cls) -> str:
        """Get formatted critical rules"""
        return "\n".join([f"- {rule}" for rule in cls.CRITICAL_RULES])


class SemanticMappings:
    """Semantic entity mappings for natural language understanding"""
    
    MAPPINGS = {
        'employee': ['employee', 'staff', 'worker', 'personnel'],
        'department': ['department', 'dept', 'division'],
        'student': ['student', 'pupil', 'learner'],
        'teacher': ['teacher', 'faculty', 'instructor', 'professor'],
        'course': ['course', 'class', 'subject'],
        'order': ['order', 'purchase', 'transaction'],
        'customer': ['customer', 'client', 'buyer'],
        'product': ['product', 'item', 'goods']
    }
    
    @classmethod
    def get_synonyms(cls, entity: str) -> List[str]:
        """Get synonyms for an entity"""
        return cls.MAPPINGS.get(entity.lower(), [entity])


class PromptExamples:
    """Example demonstrations for few-shot learning"""
    
    EXAMPLES = [
        {
            "question": "How many employees are there?",
            "schema": "table: employees [id, name, department_id, salary]",
            "sql": "SELECT COUNT(*) FROM employees",
            "explanation": "Simple count of all records"
        },
        {
            "question": "Show employee names with their department names",
            "schema": "table: employees [id, name, department_id], table: departments [id, name]",
            "sql": "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id",
            "explanation": "JOIN to combine data from related tables"
        },
        {
            "question": "Find employees with salary greater than 50000",
            "schema": "table: employees [id, name, salary]",
            "sql": "SELECT * FROM employees WHERE salary > 50000",
            "explanation": "Filtering with WHERE clause"
        },
        {
            "question": "Count employees by department",
            "schema": "table: employees [id, name, department_id], table: departments [id, name]",
            "sql": "SELECT d.name, COUNT(*) FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.id, d.name",
            "explanation": "Aggregation with GROUP BY"
        },
        {
            "question": "Show top 5 highest paid employees",
            "schema": "table: employees [id, name, salary]",
            "sql": "SELECT * FROM employees ORDER BY salary DESC LIMIT 5",
            "explanation": "Ordering and limiting results"
        },
        {
            "question": "Find customers whose email contains gmail",
            "schema": "table: customers [id, name, email]",
            "sql": "SELECT * FROM customers WHERE email LIKE '%gmail%'",
            "explanation": "Pattern matching with LIKE"
        },
        {
            "question": "What is the average salary by department?",
            "schema": "table: employees [id, name, department_id, salary], table: departments [id, name]",
            "sql": "SELECT d.name, AVG(e.salary) FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.id, d.name",
            "explanation": "Average calculation with grouping"
        }
    ]
    
    @classmethod
    def format_examples(cls, max_examples: int = 5, include_explanation: bool = True) -> str:
        """Format examples as text"""
        formatted = "## Examples:\n\n"
        
        for i, example in enumerate(cls.EXAMPLES[:max_examples], 1):
            formatted += f"**Example {i}:**\n"
            formatted += f"Question: \"{example['question']}\"\n"
            formatted += f"Schema: {example['schema']}\n"
            formatted += f"SQL: `{example['sql']}`\n"
            
            if include_explanation and example.get('explanation'):
                formatted += f"Explanation: {example['explanation']}\n"
            
            formatted += "\n"
        
        return formatted


class BasePromptTemplate:
    """Base class for prompt templates"""
    
    def __init__(self):
        self.patterns = SQLPatterns()
        self.rules = SQLValidationRules()
        self.examples = PromptExamples()
    
    def format(self, schema: str, question: str, **kwargs) -> str:
        """
        Format the template with provided data
        
        Args:
            schema: Database schema description
            question: Natural language question
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement format()")


class BasicPromptTemplate(BasePromptTemplate):
    """Basic SQL generation template"""
    
    TEMPLATE = """You are an expert SQL developer. Convert natural language to SQL.

Database Schema:
{schema}

CRITICAL RULES (MUST FOLLOW):
{critical_rules}

SINGLE TABLE QUERY FORMAT:
- Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1

Question: {question}

SQL Query:"""
    
    def format(self, schema: str, question: str, **kwargs) -> str:
        return self.TEMPLATE.format(
            schema=schema,
            question=question,
            critical_rules=self.rules.get_critical_rules_text()
        )


class FewShotPromptTemplate(BasePromptTemplate):
    """Few-shot learning template with examples"""
    
    TEMPLATE = """You are an expert SQL developer. Convert natural language questions to SQL queries based on the provided database schema.

{examples}

Database Schema:
{schema}

CRITICAL RULES (MUST FOLLOW):
{critical_rules}

Question: {question}

SQL Query:"""
    
    def format(self, schema: str, question: str, max_examples: int = 5, **kwargs) -> str:
        return self.TEMPLATE.format(
            schema=schema,
            question=question,
            examples=self.examples.format_examples(max_examples),
            critical_rules=self.rules.get_critical_rules_text()
        )


class ChainOfThoughtPromptTemplate(BasePromptTemplate):
    """Chain-of-thought reasoning template"""
    
    TEMPLATE = """You are an expert SQL developer. Convert natural language questions to SQL queries using step-by-step reasoning.

Follow this reasoning process:
1. **Analyze the question**: Identify what information is being requested
2. **Identify required tables**: Determine which tables contain the needed data
3. **Identify required columns**: Determine which columns to select or filter on
4. **Determine relationships**: Identify any JOIN conditions needed
5. **Apply filters**: Determine WHERE conditions from the question
6. **Apply aggregations**: Determine if GROUP BY, HAVING, or aggregate functions are needed
7. **Apply ordering/limiting**: Determine if ORDER BY or LIMIT clauses are needed
8. **Construct SQL**: Build the final SQL query

Database Schema:
{schema}

Question: {question}

Step-by-step reasoning:
1. **Analyze the question**: 
2. **Identify required tables**: 
3. **Identify required columns**: 
4. **Determine relationships**: 
5. **Apply filters**: 
6. **Apply aggregations**: 
7. **Apply ordering/limiting**: 
8. **Construct SQL**: 

CRITICAL RULES (MUST FOLLOW):
{critical_rules}

Final SQL Query:"""
    
    def format(self, schema: str, question: str, **kwargs) -> str:
        return self.TEMPLATE.format(
            schema=schema,
            question=question,
            critical_rules=self.rules.get_critical_rules_text()
        )


class RuleBasedPromptTemplate(BasePromptTemplate):
    """Rule-based validation template"""
    
    TEMPLATE = """You are an expert SQL query generator with strict validation rules. Generate syntactically correct and logically sound SQL queries.

## VALIDATION RULES:
{rules}

## COMMON QUESTION PATTERNS:
{patterns}

## SCHEMA VALIDATION CHECKLIST:
- ✓ All table names exist in the provided schema
- ✓ All column names belong to their respective tables
- ✓ JOIN conditions use proper foreign key relationships
- ✓ Aggregate functions follow GROUP BY rules
- ✓ WHERE clauses use appropriate operators and data types
- ✓ String values are properly quoted
- ✓ Numeric comparisons use correct operators

## ERROR HANDLING:
- If a question cannot be answered with the given schema, return: "ERROR: INSUFFICIENT_SCHEMA"
- If table/column names are ambiguous, use the most likely match from the schema
- If multiple interpretations are possible, choose the most straightforward one

Database Schema:
{schema}

Question: {question}

Validation Check:
1. Required tables: [List tables needed]
2. Required columns: [List columns needed]  
3. JOIN conditions: [List any JOINs needed]
4. Filter conditions: [List WHERE conditions]
5. Single table check: [If single table, no aliases allowed]
6. Aggregation needs: [List GROUP BY/HAVING needs]

SQL Query:"""
    
    def format(self, schema: str, question: str, **kwargs) -> str:
        return self.TEMPLATE.format(
            schema=schema,
            question=question,
            rules=self.rules.get_rules_text(),
            patterns=self.patterns.get_patterns_text()
        )


class EnhancedPromptTemplate(BasePromptTemplate):
    """Enhanced template combining multiple techniques"""
    
    TEMPLATE = """# Expert SQL Query Generator with Validation

You are an elite SQL developer with deep knowledge of database systems and query optimization. Generate accurate, efficient, and syntactically correct SQL queries.

## CORE CAPABILITIES:
- Schema-aware query generation
- Automatic error detection and correction
- Optimization for performance
- Support for complex queries (JOINs, subqueries, aggregations)

## VALIDATION RULES:
{rules}

## COMMON QUESTION PATTERNS:
{patterns}

{examples}

## QUERY CONSTRUCTION PROCESS:
1. **Question Analysis**: Parse the natural language to identify intent
2. **Schema Mapping**: Map question elements to database schema
3. **Query Planning**: Design the optimal query structure
4. **Validation**: Ensure all rules and constraints are satisfied
5. **Generation**: Produce the final SQL query

## ERROR PREVENTION:
- Validate table and column existence before generation
- Ensure proper JOIN syntax and conditions
- Apply correct GROUP BY rules for aggregations
- Use appropriate data types and operators
- Handle edge cases (NULL values, empty results)

## CRITICAL RULES (MUST FOLLOW):
{critical_rules}

## OUTPUT FORMAT:
Generate only the SQL query without explanations unless specifically requested.
Ensure the query is ready for immediate execution.

Database Schema:
{schema}

Question: {question}

SQL Query:"""
    
    def format(self, schema: str, question: str, max_examples: int = 3, **kwargs) -> str:
        return self.TEMPLATE.format(
            schema=schema,
            question=question,
            rules=self.rules.get_rules_text(),
            patterns=self.patterns.get_patterns_text(),
            examples=self.examples.format_examples(max_examples),
            critical_rules=self.rules.get_critical_rules_text()
        )


class StepByStepPromptTemplate(BasePromptTemplate):
    """Step-by-step analysis template"""
    
    TEMPLATE = """You are an expert SQL developer. Break down the question into steps and generate SQL accordingly.

Database Schema:
{schema}

Question: {question}

Analysis:

**Step 1: Question Decomposition**
- Main request: [What is being asked?]
- Entity focus: [What entities are involved?]
- Conditions: [What conditions/filters are mentioned?]
- Output format: [What should be returned?]

**Step 2: Schema Mapping**
- Primary tables: [Which tables contain the main data?]
- Supporting tables: [Which tables provide additional context?]
- Key columns: [Which columns are needed for selection/filtering?]
- Relationships: [How are tables connected?]

**Step 3: Query Structure Planning**
- SELECT clause: [What columns to return?]
- FROM clause: [Which tables to use?]
- JOIN operations: [What JOINs are needed?]
- WHERE conditions: [What filters to apply?]
- GROUP BY needs: [Any grouping required?]
- ORDER BY needs: [Any sorting required?]
- LIMIT needs: [Any result limiting?]

**Step 4: SQL Construction**
Based on the analysis above, the SQL query is:
```sql
[GENERATED SQL QUERY]
```

**Step 5: Validation Check**
- ✓ All tables exist in schema
- ✓ All columns exist in their tables
- ✓ JOIN conditions are correct
- ✓ Syntax is valid
- ✓ Logic matches the question

CRITICAL RULES (MUST FOLLOW):
{critical_rules}"""
    
    def format(self, schema: str, question: str, **kwargs) -> str:
        return self.TEMPLATE.format(
            schema=schema,
            question=question,
            critical_rules=self.rules.get_critical_rules_text()
        )