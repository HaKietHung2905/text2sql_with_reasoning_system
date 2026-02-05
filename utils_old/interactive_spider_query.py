import os
import json
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from sentence_transformers import SentenceTransformer

class InteractiveSpiderQuery:
    def __init__(self, data_dir="./spider_data", persist_dir="./chromadb"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with better compatibility
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            print("Using persistent ChromaDB client")
        except Exception as e:
            print(f"Persistent client failed: {e}")
            print("Using in-memory ChromaDB client")
            self.client = chromadb.Client()
        
        # Initialize sentence transformer
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Collections
        self.schema_collection = None
        self.question_collection = None
        self.sql_collection = None
        
        # Data storage
        self.train_data = None
        self.dev_data = None
        self.tables_data = None
        
        # Load existing collections or create new ones
        self.load_or_create_collections()
    
    def load_or_create_collections(self):
        """Load existing collections or create new ones if they don't exist"""
        try:
            # Try to get existing collections
            self.schema_collection = self.client.get_collection("spider_schemas")
            self.question_collection = self.client.get_collection("spider_questions")
            self.sql_collection = self.client.get_collection("spider_sql")
            
            print("Loaded existing ChromaDB collections")
            
            # Check if collections have data
            schema_count = self.schema_collection.count()
            question_count = self.question_collection.count()
            sql_count = self.sql_collection.count()
            
            print(f"Collection sizes: Schemas={schema_count}, Questions={question_count}, SQL={sql_count}")
            
            if schema_count == 0 or question_count == 0 or sql_count == 0:
                print("Collections are empty. Need to populate with data.")
                return False
            
            return True
            
        except Exception as e:
            print(f"Collections not found or error loading: {e}")
            print("You need to run the setup script first to populate ChromaDB")
            return False
    
    def query_similar_questions(self, user_question, n_results=5, min_similarity=0.3):
        """Find similar questions and return SQL queries with similarity scores"""
        
        if not self.question_collection:
            return {"error": "Question collection not available. Run setup first."}
        
        # Create embedding for user question
        query_embedding = self.model.encode(user_question)
        
        # Search for similar questions
        results = self.question_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['metadatas'][0]:
            return {"error": "No similar questions found"}
        
        # Process results
        similar_queries = []
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            similarity = 1 - distance  # Convert distance to similarity
            
            # Filter by minimum similarity
            if similarity >= min_similarity:
                similar_queries.append({
                    'rank': i + 1,
                    'question': metadata.get('question', ''),
                    'sql_query': metadata.get('sql', ''),
                    'database': metadata.get('db_id', 'unknown'),
                    'similarity_score': round(similarity, 4),
                    'split': metadata.get('split', 'unknown')
                })
        
        return {
            "user_question": user_question,
            "total_results": len(similar_queries),
            "results": similar_queries
        }
    
    def query_similar_sql(self, user_question, n_results=5, min_similarity=0.3):
        """Find similar SQL queries directly"""
        
        if not self.sql_collection:
            return {"error": "SQL collection not available. Run setup first."}
        
        # Create embedding for user question
        query_embedding = self.model.encode(user_question)
        
        # Search for similar SQL queries
        results = self.sql_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['metadatas'][0]:
            return {"error": "No similar SQL queries found"}
        
        # Process results
        similar_sql = []
        for i, (document, metadata, distance) in enumerate(zip(
            results['documents'][0], results['metadatas'][0], results['distances'][0]
        )):
            similarity = 1 - distance
            
            if similarity >= min_similarity:
                similar_sql.append({
                    'rank': i + 1,
                    'sql_query': document,
                    'original_question': metadata.get('question', ''),
                    'database': metadata.get('db_id', 'unknown'),
                    'similarity_score': round(similarity, 4),
                    'split': metadata.get('split', 'unknown')
                })
        
        return {
            "user_question": user_question,
            "total_results": len(similar_sql),
            "results": similar_sql
        }
    
    def find_relevant_schemas(self, user_question, n_results=3):
        """Find relevant database schemas for the question"""
        
        if not self.schema_collection:
            return {"error": "Schema collection not available. Run setup first."}
        
        query_embedding = self.model.encode(user_question)
        
        results = self.schema_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['metadatas'][0]:
            return {"error": "No relevant schemas found"}
        
        relevant_schemas = []
        for i, (document, metadata, distance) in enumerate(zip(
            results['documents'][0], results['metadatas'][0], results['distances'][0]
        )):
            similarity = 1 - distance
            relevant_schemas.append({
                'rank': i + 1,
                'database': metadata.get('db_id', 'unknown'),
                'schema': document,
                'tables': metadata.get('num_tables', 0),
                'columns': metadata.get('num_columns', 0),
                'similarity_score': round(similarity, 4)
            })
        
        return {
            "user_question": user_question,
            "relevant_schemas": relevant_schemas
        }
    
    def comprehensive_query(self, user_question, n_results=3, min_similarity=0.3):
        """Get comprehensive results: similar questions, SQL, and schemas"""
        
        return {
            "user_question": user_question,
            "similar_questions": self.query_similar_questions(user_question, n_results, min_similarity),
            "similar_sql": self.query_similar_sql(user_question, n_results, min_similarity),
            "relevant_schemas": self.find_relevant_schemas(user_question, n_results)
        }
    
    def interactive_session(self):
        """Start an interactive session for querying"""
        
        print("\n" + "="*60)
        print("Interactive Spider Query System")
        print("="*60)
        print("Enter your questions to find similar SQL queries from the Spider dataset")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the session")
        print("  'help' - Show this help message")
        print("  'schema <question>' - Find relevant database schemas")
        print("  'sql <question>' - Find similar SQL queries directly")
        print("  '<question>' - Find similar questions and SQL (comprehensive)")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nEnter your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit the session")
                    print("  'schema <question>' - Find relevant database schemas")
                    print("  'sql <question>' - Find similar SQL queries directly")
                    print("  '<question>' - Find similar questions and SQL")
                    continue
                
                if not user_input:
                    print("Please enter a question.")
                    continue
                
                # Parse command
                if user_input.lower().startswith('schema '):
                    question = user_input[7:].strip()
                    # Remove angle brackets if present
                    if question.startswith('<') and question.endswith('>'):
                        question = question[1:-1].strip()
                    results = self.find_relevant_schemas(question)
                    self._print_schema_results(results)
                
                elif user_input.lower().startswith('sql '):
                    question = user_input[4:].strip()
                    # Remove angle brackets if present
                    if question.startswith('<') and question.endswith('>'):
                        question = question[1:-1].strip()
                    results = self.query_similar_sql(question)
                    self._print_sql_results(results)
                
                else:
                    # Comprehensive query
                    question = user_input.strip()
                    # Remove angle brackets if present
                    if question.startswith('<') and question.endswith('>'):
                        question = question[1:-1].strip()
                    results = self.query_similar_questions(question)
                    self._print_question_results(results)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_question_results(self, results):
        """Print formatted results for question queries"""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\nQuestion: {results['user_question']}")
        print(f"Found {results['total_results']} similar questions:")
        print("-" * 50)
        
        for result in results['results']:
            print(f"\n{result['rank']}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Question: {result['question']}")
            print(f"   SQL: {result['sql_query']}")
            print(f"   Database: {result['database']}")
    
    def _print_sql_results(self, results):
        """Print formatted results for SQL queries"""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\nQuestion: {results['user_question']}")
        print(f"Found {results['total_results']} similar SQL queries:")
        print("-" * 50)
        
        for result in results['results']:
            print(f"\n{result['rank']}. Similarity: {result['similarity_score']:.3f}")
            print(f"   SQL: {result['sql_query']}")
            print(f"   Original Question: {result['original_question']}")
            print(f"   Database: {result['database']}")
    
    def show_all_schemas(self):
        """Show all available database schemas"""
        if not self.schema_collection:
            print("Schema collection not available. Run setup first.")
            return
        
        print("\nAll Available Database Schemas:")
        print("="*80)
        
        try:
            # Get all schemas
            all_schemas = self.schema_collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_schemas['documents']:
                print("No schemas found in the collection.")
                return
            
            for i, (document, metadata) in enumerate(zip(all_schemas['documents'], all_schemas['metadatas'])):
                db_id = metadata.get('db_id', 'unknown')
                num_tables = metadata.get('num_tables', 0)
                num_columns = metadata.get('num_columns', 0)
                
                print(f"\n{i+1}. Database: {db_id}")
                print(f"   Tables: {num_tables}, Columns: {num_columns}")
                print(f"   Schema:")
                print("   " + "-"*60)
                
                schema_lines = document.split('\n')
                for line in schema_lines:
                    if line.strip():
                        print(f"   {line}")
                
                print("   " + "-"*60)
                
        except Exception as e:
            print(f"Error retrieving schemas: {e}")
    
    def _print_schema_results(self, results):
        """Print formatted results for schema queries"""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\nQuestion: {results['user_question']}")
        print("Relevant database schemas:")
        print("-" * 50)
        
        for schema in results['relevant_schemas']:
            print(f"\n{schema['rank']}. Database: {schema['database']} (Similarity: {schema['similarity_score']:.3f})")
            print(f"   Tables: {schema['tables']}, Columns: {schema['columns']}")
            print(f"   Complete Schema:")
            print("   " + "="*60)
            
            # Show the complete schema with proper formatting
            schema_lines = schema['schema'].split('\n')
            for line in schema_lines:
                if line.strip():  # Only print non-empty lines
                    print(f"   {line}")
            
            print("   " + "="*60)
            print()  # Add spacing between schemas

def main():
    """Main function to run the interactive query system"""
    
    print("Starting Interactive Spider Query System...")
    
    # Initialize the query system
    query_system = InteractiveSpiderQuery()
    
    # Check if collections are available
    collections_ready = query_system.load_or_create_collections()
    
    if not collections_ready:
        print("\nError: ChromaDB collections are not available or empty.")
        print("Please run the setup script first to populate the database:")
        print("  python spider_chromadb_integration.py")
        return
    
    # Start interactive session
    query_system.interactive_session()

# Example usage functions
def example_queries():
    """Show some example queries"""
    query_system = InteractiveSpiderQuery()
    
    example_questions = [
        "How many restaurants are there?",
        "What are the names of all students?",
        "Show me the total sales",
        "Find customers with orders"
    ]
    
    print("Example Queries:")
    print("=" * 50)
    
    for question in example_questions:
        print(f"\nQuery: {question}")
        results = query_system.query_similar_questions(question, n_results=2)
        
        if "error" not in results:
            for result in results['results'][:2]:  # Show top 2
                print(f"  Similarity: {result['similarity_score']:.3f}")
                print(f"  SQL: {result['sql_query']}")
                print(f"  Database: {result['database']}")
        else:
            print(f"  Error: {results['error']}")

if __name__ == "__main__":
     main()