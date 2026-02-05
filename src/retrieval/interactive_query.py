"""
Interactive query system for Spider dataset.
Allows users to search for similar questions, SQL, and schemas.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import SpiderRetriever
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class InteractiveQuerySystem:
    """Interactive command-line interface for querying Spider dataset"""
    
    def __init__(self, persist_dir: str = "./data/embeddings/chroma_db"):
        """
        Initialize interactive system
        
        Args:
            persist_dir: ChromaDB persist directory
        """
        self.retriever = SpiderRetriever(persist_dir)
        self.running = False
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 70)
        print(" " * 15 + "Interactive Spider Query System")
        print("=" * 70)
        print("\nSearch for similar SQL queries from the Spider dataset")
        print("\nCommands:")
        print("  <question>              - Find similar questions and SQL")
        print("  sql <question>          - Find similar SQL queries directly")
        print("  schema <question>       - Find relevant database schemas")
        print("  all-schemas             - Show all available schemas")
        print("  help                    - Show this help message")
        print("  quit, exit, q           - Exit the system")
        print("-" * 70)
    
    def print_help(self):
        """Print help message"""
        print("\nAvailable Commands:")
        print("  <question>              - Comprehensive search (questions + SQL + schemas)")
        print("  sql <question>          - Search SQL queries only")
        print("  schema <question>       - Search database schemas only")
        print("  all-schemas             - Display all available database schemas")
        print("  help                    - Show this help")
        print("  quit, exit, q           - Exit")
        print("\nExamples:")
        print('  "How many restaurants are there?"')
        print('  sql "count all employees"')
        print('  schema "student courses database"')
    
    def clean_input(self, text: str) -> str:
        """
        Clean user input
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        text = text.strip()
        # Remove angle brackets if present
        if text.startswith('<') and text.endswith('>'):
            text = text[1:-1].strip()
        return text
    
    def print_question_results(self, results: Dict[str, Any]):
        """Print formatted question search results"""
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        query = results.get('query', '')
        total = results.get('total_results', 0)
        items = results.get('results', [])
        
        print(f"\n🔍 Query: {query}")
        print(f"📊 Found {total} similar question(s)")
        print("-" * 70)
        
        if not items:
            print("No results found. Try a different query or lower the similarity threshold.")
            return
        
        for item in items:
            print(f"\n{item['rank']}. Similarity: {item['similarity_score']:.3f} ⭐")
            print(f"   Question: {item['question']}")
            print(f"   SQL: {item['sql_query']}")
            print(f"   Database: {item['database']} | Split: {item['split']}")
    
    def print_sql_results(self, results: Dict[str, Any]):
        """Print formatted SQL search results"""
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        query = results.get('query', '')
        total = results.get('total_results', 0)
        items = results.get('results', [])
        
        print(f"\n🔍 Query: {query}")
        print(f"📊 Found {total} similar SQL quer(ies)")
        print("-" * 70)
        
        if not items:
            print("No results found. Try a different query or lower the similarity threshold.")
            return
        
        for item in items:
            print(f"\n{item['rank']}. Similarity: {item['similarity_score']:.3f} ⭐")
            print(f"   SQL: {item['sql_query']}")
            print(f"   Original Question: {item['original_question']}")
            print(f"   Database: {item['database']} | Split: {item['split']}")
    
    def print_schema_results(self, results: Dict[str, Any]):
        """Print formatted schema search results"""
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        query = results.get('query', '')
        schemas = results.get('relevant_schemas', [])
        
        print(f"\n🔍 Query: {query}")
        print(f"📊 Found {len(schemas)} relevant schema(s)")
        print("-" * 70)
        
        if not schemas:
            print("No schemas found.")
            return
        
        for schema in schemas:
            print(f"\n{schema['rank']}. Database: {schema['database']} "
                  f"(Similarity: {schema['similarity_score']:.3f} ⭐)")
            print(f"   Tables: {schema['tables']}, Columns: {schema['columns']}")
            print(f"\n   Schema:")
            print("   " + "─" * 65)
            
            for line in schema['schema'].split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            print("   " + "─" * 65)
    
    def print_all_schemas(self):
        """Print all available schemas"""
        print("\n📚 All Available Database Schemas")
        print("=" * 70)
        
        schemas = self.retriever.get_all_schemas()
        
        if not schemas:
            print("No schemas found. Run build_chromadb.py first.")
            return
        
        for i, schema in enumerate(schemas, 1):
            print(f"\n{i}. Database: {schema['database']}")
            print(f"   Tables: {schema['tables']}, Columns: {schema['columns']}")
            print("   " + "─" * 65)
            
            for line in schema['schema'].split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            print("   " + "─" * 65)
    
    def process_command(self, user_input: str):
        """
        Process user command
        
        Args:
            user_input: User input string
        """
        user_input = user_input.strip()
        
        # Check for quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            self.running = False
            return
        
        # Check for help
        if user_input.lower() == 'help':
            self.print_help()
            return
        
        # Check for empty input
        if not user_input:
            print("⚠️  Please enter a question or command.")
            return
        
        # Check for all-schemas command
        if user_input.lower() == 'all-schemas':
            self.print_all_schemas()
            return
        
        # Parse and execute command
        try:
            if user_input.lower().startswith('schema '):
                # Schema search
                question = self.clean_input(user_input[7:])
                results = self.retriever.retrieve_relevant_schemas(question)
                self.print_schema_results(results)
            
            elif user_input.lower().startswith('sql '):
                # SQL search
                question = self.clean_input(user_input[4:])
                results = self.retriever.retrieve_similar_sql(question)
                self.print_sql_results(results)
            
            else:
                # Question search (default)
                question = self.clean_input(user_input)
                results = self.retriever.retrieve_similar_questions(question)
                self.print_question_results(results)
        
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            print(f"\n❌ Error: {e}")
    
    def run(self):
        """Run the interactive session"""
        self.print_welcome()
        self.running = True
        
        while self.running:
            try:
                user_input = input("\n💬 Enter your question: ").strip()
                self.process_command(user_input)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\n❌ Error: {e}")


def show_examples():
    """Show example queries"""
    print("\n" + "=" * 70)
    print(" " * 20 + "Example Queries")
    print("=" * 70)
    
    retriever = SpiderRetriever()
    
    example_questions = [
        "How many restaurants are there?",
        "What are the names of all students?",
        "Show me the total sales",
        "Find customers with orders"
    ]
    
    for question in example_questions:
        print(f"\n🔍 Query: {question}")
        results = retriever.retrieve_similar_questions(question, n_results=2)
        
        if "error" not in results and results.get('results'):
            for result in results['results'][:2]:
                print(f"  ⭐ Similarity: {result['similarity_score']:.3f}")
                print(f"     SQL: {result['sql_query']}")
                print(f"     Database: {result['database']}")
        else:
            print(f"  ❌ No results found")
        
        print("-" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive query system for Spider dataset'
    )
    
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./data/embeddings/chroma_db',
        help='ChromaDB persist directory'
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show example queries and exit'
    )
    
    args = parser.parse_args()
    
    try:
        if args.examples:
            show_examples()
            return 0
        
        # Run interactive system
        system = InteractiveQuerySystem(args.persist_dir)
        system.run()
        return 0
    
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've run: python scripts/build_chromadb.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())