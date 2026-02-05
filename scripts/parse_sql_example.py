"""
Example script demonstrating SQL parsing.
Shows how to use the SQL parser with various complexity levels.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sql_schema import Schema, get_schema_from_sqlite, load_schema
from src.data.sql_parser import SQLParser, parse_sql
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def print_parsed_structure(parsed: dict, title: str = "Parsed SQL"):
    """Pretty print parsed SQL structure"""
    logger.info(f"\n{title}")
    logger.info("=" * 60)
    logger.info(json.dumps(parsed, indent=2, default=str))
    logger.info("=" * 60)


def example_parse_simple():
    """Example 1: Parse simple SELECT query"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Simple SELECT Query")
    logger.info("=" * 60)
    
    # Create schema
    schema_dict = {
        'employees': ['id', 'name', 'salary', 'department_id'],
        'departments': ['id', 'name']
    }
    schema = Schema(schema_dict)
    
    # Parse query
    query = "SELECT name, salary FROM employees WHERE salary > 50000"
    
    logger.info(f"\nQuery: {query}")
    logger.info(f"Schema: {list(schema_dict.keys())}")
    
    try:
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "Parsed Structure")
        
        # Show specific parts
        logger.info("\nKey Components:")
        logger.info(f"  SELECT: {parsed['select']}")
        logger.info(f"  FROM: {parsed['from']}")
        logger.info(f"  WHERE: {parsed['where']}")
        
    except Exception as e:
        logger.error(f"Parse error: {e}")


def example_parse_aggregation():
    """Example 2: Parse query with aggregation"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Aggregation Query")
    logger.info("=" * 60)
    
    schema_dict = {
        'employees': ['id', 'name', 'salary', 'department_id'],
        'departments': ['id', 'name']
    }
    schema = Schema(schema_dict)
    
    query = "SELECT COUNT(*), AVG(salary) FROM employees"
    
    logger.info(f"\nQuery: {query}")
    
    try:
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "Parsed with Aggregations")
        
        logger.info("\nAggregation Functions Detected:")
        for agg_id, val_unit in parsed['select'][1]:
            logger.info(f"  Aggregation ID: {agg_id}")
            
    except Exception as e:
        logger.error(f"Parse error: {e}")


def example_parse_join():
    """Example 3: Parse query with JOIN"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: JOIN Query")
    logger.info("=" * 60)
    
    schema_dict = {
        'employees': ['id', 'name', 'salary', 'department_id'],
        'departments': ['id', 'name']
    }
    schema = Schema(schema_dict)
    
    query = """
    SELECT e.name, d.name 
    FROM employees e 
    JOIN departments d ON e.department_id = d.id 
    WHERE e.salary > 50000
    """
    
    logger.info(f"\nQuery: {query.strip()}")
    
    try:
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "Parsed JOIN Query")
        
        logger.info("\nJOIN Information:")
        logger.info(f"  Table units: {parsed['from']['table_units']}")
        logger.info(f"  Join conditions: {parsed['from']['conds']}")
        
    except Exception as e:
        logger.error(f"Parse error: {e}")


def example_parse_complex():
    """Example 4: Parse complex SQL with multiple clauses"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Complex Query")
    logger.info("=" * 60)
    
    schema_dict = {
        'employees': ['id', 'name', 'salary', 'department_id'],
        'departments': ['id', 'name']
    }
    schema = Schema(schema_dict)
    
    query = """
    SELECT d.name, COUNT(*) as emp_count, AVG(e.salary) as avg_salary
    FROM employees e 
    JOIN departments d ON e.department_id = d.id 
    WHERE e.salary > 50000 
    GROUP BY d.id, d.name 
    HAVING COUNT(*) > 5
    ORDER BY COUNT(*) DESC 
    LIMIT 10
    """
    
    logger.info(f"\nQuery: {query.strip()}")
    
    try:
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "Complex Query Structure")
        
        logger.info("\nQuery Complexity Analysis:")
        logger.info(f"  Has JOIN: {len(parsed['from']['table_units']) > 1}")
        logger.info(f"  Has WHERE: {len(parsed['where']) > 0}")
        logger.info(f"  Has GROUP BY: {len(parsed['groupBy']) > 0}")
        logger.info(f"  Has HAVING: {len(parsed['having']) > 0}")
        logger.info(f"  Has ORDER BY: {len(parsed['orderBy'][1]) > 0}")
        logger.info(f"  Has LIMIT: {parsed['limit'] is not None}")
        
    except Exception as e:
        logger.error(f"Parse error: {e}")


def example_parse_subquery():
    """Example 5: Parse query with subquery"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Subquery")
    logger.info("=" * 60)
    
    schema_dict = {
        'employees': ['id', 'name', 'salary', 'department_id'],
        'departments': ['id', 'name']
    }
    schema = Schema(schema_dict)
    
    query = """
    SELECT name, salary 
    FROM employees 
    WHERE salary > (SELECT AVG(salary) FROM employees)
    """
    
    logger.info(f"\nQuery: {query.strip()}")
    
    try:
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "Query with Subquery")
        
        logger.info("\nSubquery Detection:")
        if parsed['where']:
            logger.info(f"  WHERE clause contains conditions: {len(parsed['where'])}")
        
    except Exception as e:
        logger.error(f"Parse error: {e}")


def example_parse_union():
    """Example 6: Parse UNION query"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: UNION Query")
    logger.info("=" * 60)
    
    schema_dict = {
        'employees': ['id', 'name', 'salary'],
        'contractors': ['id', 'name', 'rate']
    }
    schema = Schema(schema_dict)
    
    query = """
    SELECT name FROM employees 
    UNION 
    SELECT name FROM contractors
    """
    
    logger.info(f"\nQuery: {query.strip()}")
    
    try:
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "UNION Query Structure")
        
        logger.info("\nSet Operations:")
        logger.info(f"  Has UNION: {parsed['union'] is not None}")
        logger.info(f"  Has INTERSECT: {parsed['intersect'] is not None}")
        logger.info(f"  Has EXCEPT: {parsed['except'] is not None}")
        
    except Exception as e:
        logger.error(f"Parse error: {e}")


def example_with_spider_db():
    """Example 7: Parse SQL with actual Spider database"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: Parse with Spider Database")
    logger.info("=" * 60)
    
    # Try to load from Spider dataset
    db_path = "./data/raw/spider/database/concert_singer/concert_singer.sqlite"
    
    if not Path(db_path).exists():
        logger.warning(f"Database not found: {db_path}")
        logger.info("Skipping this example. Run build_chromadb.py first.")
        return
    
    try:
        # Load schema from database
        schema = load_schema(db_path)
        
        logger.info(f"Loaded schema from: {db_path}")
        logger.info(f"Tables: {list(schema.schema.keys())}")
        
        # Parse query
        query = "SELECT COUNT(*) FROM singer WHERE age > 25"
        
        logger.info(f"\nQuery: {query}")
        
        parsed = parse_sql(query, schema)
        print_parsed_structure(parsed, "Parsed from Real Database")
        
    except Exception as e:
        logger.error(f"Error: {e}")


def example_batch_parsing():
    """Example 8: Batch parse multiple queries"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 8: Batch Parsing")
    logger.info("=" * 60)
    
    schema_dict = {
        'students': ['id', 'name', 'age', 'gpa'],
        'courses': ['id', 'name', 'credits'],
        'enrollments': ['student_id', 'course_id', 'grade']
    }
    schema = Schema(schema_dict)
    
    queries = [
        "SELECT * FROM students",
        "SELECT name FROM students WHERE age > 20",
        "SELECT COUNT(*) FROM students",
        "SELECT name, AVG(gpa) FROM students GROUP BY name",
        "SELECT s.name, c.name FROM students s JOIN enrollments e ON s.id = e.student_id JOIN courses c ON e.course_id = c.id"
    ]
    
    logger.info(f"\nParsing {len(queries)} queries...")
    
    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{i}. {query}")
        try:
            parsed = parse_sql(query, schema)
            results.append((query, parsed, None))
            logger.info("   ✓ Parsed successfully")
        except Exception as e:
            results.append((query, None, str(e)))
            logger.error(f"   ✗ Parse failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Batch Parsing Summary:")
    logger.info(f"  Total: {len(queries)}")
    logger.info(f"  Success: {sum(1 for _, p, _ in results if p is not None)}")
    logger.info(f"  Failed: {sum(1 for _, p, _ in results if p is None)}")


def example_error_handling():
    """Example 9: Error handling"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 9: Error Handling")
    logger.info("=" * 60)
    
    schema_dict = {
        'employees': ['id', 'name', 'salary']
    }
    schema = Schema(schema_dict)
    
    # Invalid queries
    invalid_queries = [
        "SELECT FROM employees",  # Missing columns
        "SELECT * FORM employees",  # Typo in FROM
        "SELECT invalid_column FROM employees",  # Non-existent column
        "SELECT * FROM invalid_table",  # Non-existent table
    ]
    
    logger.info("\nTesting error handling with invalid queries:")
    
    for query in invalid_queries:
        logger.info(f"\nQuery: {query}")
        try:
            parsed = parse_sql(query, schema)
            logger.warning("   ⚠ Parsed (unexpected)")
        except AssertionError as e:
            logger.info(f"   ✓ Caught assertion error: {e}")
        except ValueError as e:
            logger.info(f"   ✓ Caught value error: {e}")
        except Exception as e:
            logger.info(f"   ✓ Caught error: {type(e).__name__}: {e}")


def example_parser_api():
    """Example 10: Using SQLParser API directly"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 10: SQLParser API")
    logger.info("=" * 60)
    
    schema_dict = {
        'products': ['id', 'name', 'price', 'category'],
        'orders': ['id', 'product_id', 'quantity', 'date']
    }
    schema = Schema(schema_dict)
    
    # Create parser instance
    parser = SQLParser(schema)
    
    queries = [
        "SELECT name, price FROM products WHERE price < 100",
        "SELECT category, COUNT(*) FROM products GROUP BY category",
        "SELECT p.name, SUM(o.quantity) FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.name"
    ]
    
    logger.info("\nUsing SQLParser API:")
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{i}. Query: {query}")
        try:
            parsed = parser.parse(query)
            
            # Extract specific information
            logger.info("   Clauses present:")
            logger.info(f"     SELECT: {len(parsed['select'][1])} columns")
            logger.info(f"     FROM: {len(parsed['from']['table_units'])} tables")
            logger.info(f"     WHERE: {'Yes' if parsed['where'] else 'No'}")
            logger.info(f"     GROUP BY: {'Yes' if parsed['groupBy'] else 'No'}")
            
        except Exception as e:
            logger.error(f"   Error: {e}")


def main():
    """Run all examples"""
    logger.info("=" * 70)
    logger.info(" " * 20 + "SQL Parser Examples")
    logger.info("=" * 70)
    logger.info("\nThis script demonstrates various SQL parsing capabilities")
    logger.info("using the Spider dataset SQL parser.\n")
    
    examples = [
        ("Simple SELECT", example_parse_simple),
        ("Aggregation", example_parse_aggregation),
        ("JOIN", example_parse_join),
        ("Complex Query", example_parse_complex),
        ("Subquery", example_parse_subquery),
        ("UNION", example_parse_union),
        ("Spider Database", example_with_spider_db),
        ("Batch Parsing", example_batch_parsing),
        ("Error Handling", example_error_handling),
        ("Parser API", example_parser_api),
    ]
    
    try:
        for i, (name, example_func) in enumerate(examples, 1):
            logger.info(f"\n\n{'='*70}")
            logger.info(f"Running Example {i}/{len(examples)}: {name}")
            logger.info(f"{'='*70}")
            
            try:
                example_func()
            except KeyboardInterrupt:
                logger.info("\n\nExecution interrupted by user")
                break
            except Exception as e:
                logger.error(f"Example failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("\n\n" + "=" * 70)
        logger.info("All examples completed!")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n\nExecution interrupted by user")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SQL Parser Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=range(1, 11),
        help='Run specific example (1-10)'
    )
    
    args = parser.parse_args()
    
    if args.example:
        examples = {
            1: example_parse_simple,
            2: example_parse_aggregation,
            3: example_parse_join,
            4: example_parse_complex,
            5: example_parse_subquery,
            6: example_parse_union,
            7: example_with_spider_db,
            8: example_batch_parsing,
            9: example_error_handling,
            10: example_parser_api,
        }
        
        logger.info(f"Running Example {args.example}")
        examples[args.example]()
    else:
        main()