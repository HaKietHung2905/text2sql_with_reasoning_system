"""
Seed initial reasoning strategies into ReasoningBank

This creates a set of foundational strategies based on common SQL patterns
that the system can use and evolve over time.
"""
import sys
sys.path.append('.')

from src.reasoning import ReasoningBankPipeline
from src.reasoning.strategy_distillation import ReasoningStrategy
from datetime import datetime
from pathlib import Path

def create_aggregation_strategy():
    """Strategy for COUNT/SUM/AVG queries"""
    return ReasoningStrategy(
        strategy_id="strat_agg_001",
        name="Basic Aggregation Strategy",
        pattern="aggregation",
        description="Strategy for queries requiring COUNT, SUM, AVG, MIN, or MAX operations",
        reasoning_steps=[
            "Identify the aggregation function needed (COUNT, SUM, AVG, MIN, MAX)",
            "Determine which column to aggregate",
            "Check if GROUP BY is needed for grouping results",
            "Add appropriate WHERE clauses for filtering before aggregation",
            "Consider HAVING clauses for filtering after aggregation"
        ],
        critical_rules=[
            "Always use GROUP BY when aggregating with other columns in SELECT",
            "Place aggregation functions in SELECT clause",
            "Use HAVING for conditions on aggregated values",
            "Use WHERE for conditions on non-aggregated values",
            "COUNT(*) counts all rows, COUNT(column) counts non-NULL values"
        ],
        sql_template_hints={
            "select_clause": "SELECT {group_columns}, {agg_function}({agg_column})",
            "group_by": "GROUP BY {group_columns}",
            "having": "HAVING {agg_function}({agg_column}) {condition}",
            "example": "SELECT department, COUNT(*) FROM employees GROUP BY department"
        },
        applicability={
            "keywords": [
                "how many", "count", "total", "sum", "average", "avg", 
                "maximum", "max", "minimum", "min", "highest", "lowest"
            ],
            "intent_types": ["aggregation", "statistics", "counting", "summarization"],
            "sql_patterns": ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP BY"]
        },
        common_pitfalls=[
            {
                "mistake": "Forgetting GROUP BY with non-aggregated columns",
                "fix": "Include all non-aggregated SELECT columns in GROUP BY"
            },
            {
                "mistake": "Using WHERE instead of HAVING for aggregated conditions",
                "fix": "Use HAVING for conditions on aggregated values"
            }
        ],
        success_rate=0.0,
        sample_count=0
    )

def create_join_strategy():
    """Strategy for JOIN queries"""
    return ReasoningStrategy(
        strategy_id="strat_join_001",
        name="Multi-Table Join Strategy",
        pattern="join",
        description="Strategy for queries requiring data from multiple tables",
        reasoning_steps=[
            "Identify all tables mentioned in the question",
            "Determine the relationship between tables (foreign keys)",
            "Choose appropriate JOIN type (INNER, LEFT, RIGHT)",
            "Specify JOIN conditions using foreign key relationships",
            "Select columns from appropriate tables with table prefixes"
        ],
        critical_rules=[
            "Always specify table aliases for clarity",
            "Use explicit JOIN syntax (not WHERE clause joins)",
            "Include proper JOIN conditions to avoid Cartesian products",
            "Prefix columns with table names when ambiguous",
            "Understand difference between INNER JOIN (matching only) and LEFT JOIN (all from left table)"
        ],
        sql_template_hints={
            "join_clause": "{table1} AS t1 INNER JOIN {table2} AS t2 ON t1.{fk} = t2.{pk}",
            "multiple_joins": "FROM t1 JOIN t2 ON t1.id = t2.t1_id JOIN t3 ON t2.id = t3.t2_id",
            "example": "SELECT s.name, c.course_name FROM students s JOIN enrollments e ON s.id = e.student_id JOIN courses c ON e.course_id = c.id"
        },
        applicability={
            "keywords": [
                "and", "with", "from", "in", "of", "their", "which", 
                "along with", "together with", "corresponding"
            ],
            "intent_types": ["join", "multi_table", "relationship", "association"],
            "sql_patterns": ["JOIN", "INNER JOIN", "LEFT JOIN", "FOREIGN KEY"]
        },
        common_pitfalls=[
            {
                "mistake": "Missing JOIN condition causing Cartesian product",
                "fix": "Always include ON clause with proper foreign key relationship"
            },
            {
                "mistake": "Using wrong JOIN type (INNER when LEFT needed)",
                "fix": "Use LEFT JOIN when you need all records from the first table"
            }
        ],
        success_rate=0.0,
        sample_count=0
    )

def create_filtering_strategy():
    """Strategy for WHERE clause filtering"""
    return ReasoningStrategy(
        strategy_id="strat_filter_001",
        name="Filtering and Conditions Strategy",
        pattern="filtering",
        description="Strategy for queries with WHERE clause conditions",
        reasoning_steps=[
            "Identify filtering criteria in the question",
            "Determine the columns to filter on",
            "Choose appropriate comparison operators (=, >, <, LIKE, IN)",
            "Handle NULL values with IS NULL / IS NOT NULL",
            "Combine multiple conditions with AND/OR logic"
        ],
        critical_rules=[
            "Use single quotes for string literals",
            "Use IS NULL instead of = NULL",
            "Use LIKE for pattern matching with wildcards",
            "Use IN for multiple value matching",
            "AND has higher precedence than OR (use parentheses for clarity)"
        ],
        sql_template_hints={
            "equality": "WHERE {column} = {value}",
            "comparison": "WHERE {column} > {value}",
            "pattern": "WHERE {column} LIKE '%{pattern}%'",
            "multiple": "WHERE {column1} = {value1} AND {column2} > {value2}",
            "example": "SELECT * FROM products WHERE category = 'Electronics' AND price < 1000"
        },
        applicability={
            "keywords": [
                "where", "with", "that", "which", "whose", "having",
                "greater than", "less than", "equal to", "equals",
                "like", "starting with", "ending with", "containing"
            ],
            "intent_types": ["filtering", "condition", "selection", "constraint"],
            "sql_patterns": ["WHERE", "AND", "OR", "LIKE", "IN", "BETWEEN"]
        },
        common_pitfalls=[
            {
                "mistake": "Using = NULL instead of IS NULL",
                "fix": "Always use IS NULL or IS NOT NULL for NULL checks"
            },
            {
                "mistake": "Missing quotes around string values",
                "fix": "Enclose string literals in single quotes"
            }
        ],
        success_rate=0.0,
        sample_count=0
    )

def create_ordering_strategy():
    """Strategy for ORDER BY queries"""
    return ReasoningStrategy(
        strategy_id="strat_order_001",
        name="Sorting and Ordering Strategy",
        pattern="ordering",
        description="Strategy for queries requiring sorted results",
        reasoning_steps=[
            "Identify what column(s) to sort by",
            "Determine sort direction (ascending or descending)",
            "Handle multiple sort columns (primary, secondary)",
            "Consider LIMIT clause for 'top N' queries",
            "Use column aliases if sorting by computed values"
        ],
        critical_rules=[
            "ASC is ascending (default), DESC is descending",
            "ORDER BY comes after WHERE and GROUP BY",
            "Can order by column position number in SELECT",
            "Multiple columns are ordered left to right",
            "Combine with LIMIT for 'top N' or 'bottom N' queries"
        ],
        sql_template_hints={
            "basic": "ORDER BY {column} DESC",
            "multiple": "ORDER BY {column1} DESC, {column2} ASC",
            "with_limit": "ORDER BY {column} DESC LIMIT {n}",
            "example": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 10"
        },
        applicability={
            "keywords": [
                "highest", "lowest", "top", "bottom", "largest", "smallest",
                "most", "least", "best", "worst", "first", "last",
                "sort", "order", "rank", "ascending", "descending"
            ],
            "intent_types": ["sorting", "ranking", "ordering", "top_n"],
            "sql_patterns": ["ORDER BY", "LIMIT", "DESC", "ASC"]
        },
        common_pitfalls=[
            {
                "mistake": "Forgetting DESC for 'highest' queries",
                "fix": "Use ORDER BY column DESC for highest/largest values"
            },
            {
                "mistake": "Missing LIMIT for 'top N' queries",
                "fix": "Add LIMIT clause when question asks for specific number"
            }
        ],
        success_rate=0.0,
        sample_count=0
    )

def create_subquery_strategy():
    """Strategy for subquery patterns"""
    return ReasoningStrategy(
        strategy_id="strat_subq_001",
        name="Subquery Strategy",
        pattern="subquery",
        description="Strategy for queries requiring nested subqueries",
        reasoning_steps=[
            "Identify if query needs intermediate calculation",
            "Determine if subquery should be in WHERE, FROM, or SELECT",
            "Write inner query first, then outer query",
            "Use IN/EXISTS for existence checks",
            "Use comparison operators with scalar subqueries"
        ],
        critical_rules=[
            "Subquery in WHERE must return compatible values for comparison",
            "Use IN for subqueries returning multiple values",
            "Use = for subqueries returning single value",
            "Derived tables in FROM must have aliases",
            "EXISTS is often faster than IN for large datasets"
        ],
        sql_template_hints={
            "in_where": "WHERE {column} IN (SELECT {column} FROM {table} WHERE {condition})",
            "scalar": "WHERE {column} > (SELECT AVG({column}) FROM {table})",
            "derived": "FROM (SELECT {columns} FROM {table} WHERE {condition}) AS subq",
            "example": "SELECT name FROM students WHERE gpa > (SELECT AVG(gpa) FROM students)"
        },
        applicability={
            "keywords": [
                "greater than average", "more than average", "above average",
                "who also", "who have", "that are in", "excluding",
                "except", "not in", "other than"
            ],
            "intent_types": ["subquery", "nested", "comparison", "existence"],
            "sql_patterns": ["IN", "EXISTS", "NOT IN", "scalar subquery", "derived table"]
        },
        common_pitfalls=[
            {
                "mistake": "Using = with subquery that returns multiple rows",
                "fix": "Use IN when subquery can return multiple values"
            },
            {
                "mistake": "Missing alias for derived table",
                "fix": "Always provide AS alias for subqueries in FROM"
            }
        ],
        success_rate=0.0,
        sample_count=0
    )

def seed_strategies():
    """Seed initial strategies into ReasoningBank"""
    
    print("="*80)
    print("SEEDING INITIAL STRATEGIES INTO REASONINGBANK")
    print("="*80)
    
    # Initialize pipeline
    pipeline = ReasoningBankPipeline(
        db_path="./memory/reasoning_bank.db",
        chromadb_path="./memory/chromadb"
    )
    print("\n✓ Pipeline initialized")
    
    # Create strategies
    strategies = [
        create_aggregation_strategy(),
        create_join_strategy(),
        create_filtering_strategy(),
        create_ordering_strategy(),
        create_subquery_strategy()
    ]
    
    print(f"\n📚 Created {len(strategies)} initial strategies:")
    for s in strategies:
        print(f"   - {s.name} ({s.pattern})")
    
    # Store strategies
    print("\n💾 Storing strategies in memory...")
    success_count = 0
    for strategy in strategies:
        if pipeline.memory_store.store_strategy(strategy):
            success_count += 1
            print(f"   ✓ Stored: {strategy.name}")
        else:
            print(f"   ✗ Failed: {strategy.name}")
    
    print(f"\n✅ Successfully stored {success_count}/{len(strategies)} strategies")
    
    # Verify storage
    print("\n🔍 Verifying storage...")
    stats = pipeline.memory_store.get_statistics()
    
    print(f"\n📊 Memory Store Statistics:")
    print(f"   Total strategies: {stats['total_strategies']}")
    print(f"   Active strategies: {stats['active_strategies']}")
    print(f"   ChromaDB count: {stats.get('chromadb_count', 'N/A')}")
    print(f"   Pattern distribution:")
    for pattern, count in stats['pattern_distribution'].items():
        print(f"      {pattern}: {count}")
    
    # Test retrieval
    print("\n🔎 Testing strategy retrieval...")
    
    test_queries = [
        "How many students are enrolled?",
        "What is the average salary by department?",
        "Show me employees who joined after 2020",
        "List the top 5 highest paid employees"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = pipeline.memory_store.search_strategies(query, n_results=2)
        
        if results:
            for strategy, score in results:
                print(f"      → {strategy.name} (similarity: {score:.3f})")
        else:
            print(f"      → No strategies found")
    
    print("\n" + "="*80)
    print("✅ INITIAL STRATEGIES SEEDED SUCCESSFULLY!")
    print("="*80)
    print("\nYou can now run evaluations and the system will:")
    print("  1. Retrieve these strategies for relevant queries")
    print("  2. Learn from successful applications")
    print("  3. Distill new strategies from experience")
    print("  4. Evolve existing strategies over time")
    print("="*80)

if __name__ == "__main__":
    seed_strategies()