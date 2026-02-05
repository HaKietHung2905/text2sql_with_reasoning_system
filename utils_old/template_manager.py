"""
Template manager for SQL prompt generation.
Separated from main evaluation logic for cleaner code organization.
"""

from enum import Enum

class PromptType(Enum):
    """Prompt type definitions"""
    BASIC = "basic"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RULE_BASED = "rule_based"
    ENHANCED = "enhanced"
    STEP_BY_STEP = "step_by_step"

class TemplateManager:
    """Manages SQL generation templates"""
    
    def __init__(self):
        self.common_patterns = {
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
        
        self.semantic_mappings = {
            'employee': ['employee', 'staff', 'worker', 'personnel'],
            'department': ['department', 'dept', 'division'],
            'student': ['student', 'pupil', 'learner'],
            'teacher': ['teacher', 'faculty', 'instructor', 'professor'],
            'course': ['course', 'class', 'subject'],
            'order': ['order', 'purchase', 'transaction'],
            'customer': ['customer', 'client', 'buyer'],
            'product': ['product', 'item', 'goods']
        }
        
        self.validation_rules = [
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
            "**If query use single table, query won't use table alias or column alias or field alias. Remove it"
        ]

    def get_template(self, prompt_type: PromptType) -> str:
        """Get template by type"""
        if prompt_type == PromptType.BASIC:
            return self.get_basic_template()
        elif prompt_type == PromptType.FEW_SHOT:
            return self.get_few_shot_template()
        elif prompt_type == PromptType.CHAIN_OF_THOUGHT:
            return self.get_chain_of_thought_template()
        elif prompt_type == PromptType.RULE_BASED:
            return self.get_rule_based_template()
        elif prompt_type == PromptType.ENHANCED:
            return self.get_enhanced_template()
        elif prompt_type == PromptType.STEP_BY_STEP:
            return self.get_step_by_step_template()
        else:
            return self.get_basic_template()

    def get_basic_template(self) -> str:
        """Basic SQL generation template"""
        return """You are an expert SQL developer. Convert natural language to SQL.

Database Schema:
{schema}

CRITICAL RULES (MUST FOLLOW):
- For single table queries: NO table aliases, NO column aliases, NO field aliases
- Use only tables/columns from the schema
- Generate correct SQL syntax
- Keep queries simple and efficient
- Return only SQL, no explanation

SINGLE TABLE QUERY FORMAT:
- Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1

Question: {question}

SQL Query:"""

    def get_few_shot_template(self) -> str:
        """Few-shot prompting template with examples"""
        return """You are an expert SQL developer. Convert natural language questions to SQL queries based on the provided database schema. If query use single table, query won't use table alias or column alias or field alias

## Examples:

**Example 1:**
Question: "How many employees are there?"
Schema: table: employees [id, name, department_id, salary]
SQL: `SELECT COUNT(*) FROM employees`

**Example 2:**
Question: "Show employee names with their department names"
Schema: table: employees [id, name, department_id], table: departments [id, name]
SQL: `SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id`

**Example 3:**
Question: "Find employees with salary greater than 50000"
Schema: table: employees [id, name, salary]
SQL: `SELECT * FROM employees WHERE salary > 50000`

**Example 4:**
Question: "Count employees by department"
Schema: table: employees [id, name, department_id], table: departments [id, name]
SQL: `SELECT d.name, COUNT(*) FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.id, d.name`

**Example 5:**
Question: "Show top 5 highest paid employees"
Schema: table: employees [id, name, salary]
SQL: `SELECT * FROM employees ORDER BY salary DESC LIMIT 5`

**Example 6:**
Question: "Find customers whose email contains gmail"
Schema: table: customers [id, name, email, phone]
SQL: `SELECT * FROM customers WHERE email LIKE '%gmail%'`

**Example 7:**
Question: "What is the average order amount by customer?"
Schema: table: orders [id, customer_id, amount], table: customers [id, name]
SQL: `SELECT c.name, AVG(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name`

Database Schema:
{schema}

Question: {question}

SQL Query:"""

    def get_rule_based_template(self) -> str:
        """Rule-based validation template"""
        rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(self.validation_rules)])
        patterns_text = "\n".join([f"- **\"{pattern}\"** → `{sql}`" for pattern, sql in self.common_patterns.items()])
        
        return f"""You are an expert SQL query generator with strict validation rules. Generate syntactically correct and logically sound SQL queries.

## VALIDATION RULES:
{rules_text}

## COMMON QUESTION PATTERNS:
{patterns_text}

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
{{schema}}

Question: {{question}}

Validation Check:
1. Required tables: [List tables needed]
2. Required columns: [List columns needed]  
3. JOIN conditions: [List any JOINs needed]
4. Filter conditions: [List WHERE conditions]
5. If query use single table, query won't use table alias or column alias or field alias
6. Aggregation needs: [List GROUP BY/HAVING needs]

SQL Query:"""

    def get_enhanced_template(self) -> str:
        """Enhanced template combining all techniques"""
        rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(self.validation_rules)])
        patterns_text = "\n".join([f"- **\"{pattern}\"** → `{sql}`" for pattern, sql in self.common_patterns.items()])
        
        return f"""# Expert SQL Query Generator with Validation

You are an elite SQL developer with deep knowledge of database systems and query optimization. Generate accurate, efficient, and syntactically correct SQL queries.

## CORE CAPABILITIES:
- Schema-aware query generation
- Automatic error detection and correction
- Optimization for performance
- Support for complex queries (JOINs, subqueries, aggregations)

## VALIDATION RULES:
{rules_text}

## CRITICAL RULES (MUST FOLLOW):
- For single table queries: NO table aliases, NO column aliases, NO field aliases

## SINGLE TABLE QUERY FORMAT:
- Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1


## QUESTION PATTERN RECOGNITION:
{patterns_text}

## EXAMPLE DEMONSTRATIONS:

**Example 1: Basic Count**
Question: "How many employees are there?"
Schema: table: employees [id, name, department_id, salary]
SQL: `SELECT COUNT(*) FROM employees`

**Example 2: JOIN Query**
Question: "Show employee names with their department names"
Schema: table: employees [id, name, department_id], table: departments [id, name]
SQL: `SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id`

**Example 3: Aggregation with GROUP BY**
Question: "Count employees by department"
Schema: table: employees [id, name, department_id], table: departments [id, name]
SQL: `SELECT d.name, COUNT(*) FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.id, d.name`

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

## OUTPUT FORMAT:
Generate only the SQL query without explanations unless specifically requested.
Ensure the query is ready for immediate execution.

Database Schema:
{{schema}}

Question: {{question}}

SQL Query:"""

    def get_chain_of_thought_template(self) -> str:
        """Chain-of-thought reasoning template"""
        return """You are an expert SQL developer. Convert natural language questions to SQL queries using step-by-step reasoning.

Follow this reasoning process:
1. **Analyze the question**: Identify what information is being requested
2. **Identify required tables**: Determine which tables contain the needed data
3. **Identify required columns**: Determine which columns to select or filter on
4. **Determine relationships**: Identify any JOIN conditions needed
5. **Apply filters**: Determine WHERE conditions from the question
6. **Apply aggregations**: Determine if GROUP BY, HAVING, or aggregate functions are needed
7. **Apply ordering/limiting**: Determine if ORDER BY or LIMIT clauses are needed
8. **Construct SQL**: Build the final SQL query

Example reasoning:
Question: "Show the names of employees who earn more than the average salary in their department"
Schema: table: employees [id, name, department_id, salary], table: departments [id, name]

Reasoning:
1. **Analyze**: Need employee names where salary > average salary in their department
2. **Tables**: employees (for names and salaries), departments (for department grouping)
3. **Columns**: employees.name, employees.salary, employees.department_id
4. **Relationships**: Need to group by department, so might need department info
5. **Filters**: salary > average salary per department (subquery needed)
6. **Aggregations**: Need AVG(salary) grouped by department
7. **Ordering**: Not specified
8. **SQL**: SELECT e.name FROM employees e WHERE e.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.department_id = e.department_id)

Now apply this reasoning to:

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

## CRITICAL RULES (MUST FOLLOW):
- For single table queries: NO table aliases, NO column aliases, NO field aliases

## SINGLE TABLE QUERY FORMAT:
- Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1

Final SQL Query:"""

    def get_step_by_step_template(self) -> str:
        """Step-by-step analysis template"""
        return """You are an expert SQL developer. Break down the question into steps and generate SQL accordingly.

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

## CRITICAL RULES (MUST FOLLOW):
- For single table queries: NO table aliases, NO column aliases, NO field aliases

## SINGLE TABLE QUERY FORMAT:
- Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1
"""

    def create_custom_template(self, 
                             include_examples: bool = True,
                             include_rules: bool = True,
                             include_patterns: bool = True,
                             include_reasoning: bool = False,
                             max_examples: int = 5) -> str:
        """Create a custom template with specified components"""
        
        template_parts = [
            "You are an expert SQL developer. Convert natural language questions to accurate SQL queries."
        ]
        
        if include_rules:
            rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(self.validation_rules)])
            template_parts.append(f"\n## VALIDATION RULES:\n{rules_text}")
        
        if include_patterns:
            patterns_text = "\n".join([f"- **\"{pattern}\"** → `{sql}`" for pattern, sql in self.common_patterns.items()])
            template_parts.append(f"\n## COMMON PATTERNS:\n{patterns_text}")
        
        if include_examples:
            examples_text = self._get_basic_examples(max_examples)
            template_parts.append(f"\n{examples_text}")
        
        if include_reasoning:
            template_parts.append("""
## REASONING PROCESS:
1. Analyze the question to understand the request
2. Identify required tables and columns from the schema
3. Determine necessary JOIN conditions
4. Apply appropriate WHERE filters
5. Add GROUP BY for aggregations if needed
6. Include ORDER BY and LIMIT if specified
7. Validate the final query against schema and rules

## CRITICAL RULES (MUST FOLLOW):
- For single table queries: NO table aliases, NO column aliases, NO field aliases

## SINGLE TABLE QUERY FORMAT:
- Correct: SELECT column1, COUNT(*) FROM table GROUP BY column1                             
""")
        
        template_parts.extend([
            "\nDatabase Schema:",
            "{schema}",
            "\nQuestion: {question}",
            "\nSQL Query:"
        ])
        
        return "\n".join(template_parts)

    def _get_basic_examples(self, max_examples: int = 5) -> str:
        """Get basic examples for templates"""
        examples = [
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
        
        formatted = "## Examples:\n\n"
        for i, example in enumerate(examples[:max_examples], 1):
            formatted += f"**Example {i}:**\n"
            formatted += f"Question: \"{example['question']}\"\n"
            formatted += f"Schema: {example['schema']}\n"
            formatted += f"SQL: `{example['sql']}`\n"
            if example.get('explanation'):
                formatted += f"Explanation: {example['explanation']}\n"
            formatted += "\n"
        return formatted


class SpecializedPrompts:
    """Specialized prompts for debugging and optimization"""
    
    @staticmethod
    def get_debugging_prompt() -> str:
        """Prompt for debugging and fixing SQL queries"""
        return """You are an SQL debugging expert. Fix the provided SQL query to work correctly with the given schema.

Original Question: {question}
Database Schema: {schema}
Problematic SQL: {sql}
Error Message: {error}

Please provide:
1. **Error Analysis**: What's wrong with the query?
2. **Corrected SQL**: The fixed version
3. **Explanation**: Why the fix works

Corrected SQL Query:"""
    
    @staticmethod
    def get_optimization_prompt() -> str:
        """Prompt for SQL query optimization"""
        return """You are an SQL optimization expert. Improve the given query for better performance while maintaining correctness.

Original Question: {question}
Database Schema: {schema}
Current SQL: {sql}

Please provide:
1. **Performance Analysis**: Potential issues with current query
2. **Optimized SQL**: Improved version
3. **Optimization Explanation**: What improvements were made

Optimized SQL Query:"""
    
    @staticmethod
    def get_explanation_prompt() -> str:
        """Prompt for explaining SQL queries"""
        return """You are an SQL education expert. Explain the given SQL query in clear, understandable terms.

Question: {question}
Database Schema: {schema}
SQL Query: {sql}

Please provide:
1. **Query Breakdown**: Explain each part of the query
2. **Logic Flow**: How the query works step by step
3. **Result Description**: What the query returns

Explanation:"""