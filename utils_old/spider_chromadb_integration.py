import os
import json
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Hugging Face datasets not available. Install with: pip install datasets")

class SpiderChromaDBIntegration:
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
    
    def _add_in_batches(self, collection, documents, embeddings, ids, metadatas, batch_size=1000):
        """Add data to collection in batches to avoid ChromaDB size limits"""
        total_items = len(documents)
        print(f"Adding {total_items} items in batches of {batch_size}...")
        
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            
            batch_docs = documents[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            batch_num = i // batch_size + 1
            total_batches = (total_items - 1) // batch_size + 1
            
            try:
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
                print(f"  ✓ Added batch {batch_num}/{total_batches} ({len(batch_docs)} items)")
            except Exception as e:
                print(f"  ✗ Error adding batch {batch_num}: {e}")
                # Try smaller batch size
                if batch_size > 100:
                    print(f"  Retrying batch {batch_num} with smaller batch size...")
                    self._add_in_batches(
                        collection, 
                        batch_docs, batch_embeddings, batch_ids, batch_metadatas,
                        batch_size=100
                    )
                else:
                    print(f"  Failed to add batch {batch_num}, skipping...")
    
    def load_spider_from_huggingface(self):
        """Load Spider dataset from Hugging Face"""
        if not HF_AVAILABLE:
            print("Please install datasets: pip install datasets")
            return False
        
        print("Loading Spider dataset from Hugging Face...")
        
        try:
            # Load the dataset
            dataset = load_dataset("spider")
            
            # Convert to our format
            self.train_data = []
            for item in dataset['train']:
                self.train_data.append({
                    'db_id': item['db_id'],
                    'question': item['question'],
                    'query': item['query']
                })
            
            self.dev_data = []
            for item in dataset['validation']:
                self.dev_data.append({
                    'db_id': item['db_id'],
                    'question': item['question'],
                    'query': item['query']
                })
            
            # Create tables data from the schema information
            self.tables_data = self._extract_tables_from_dataset(dataset)
            
            print(f"Loaded {len(self.train_data)} training examples")
            print(f"Loaded {len(self.dev_data)} validation examples") 
            print(f"Extracted {len(self.tables_data)} database schemas")
            
            return True
            
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            return False
    
    def _extract_tables_from_dataset(self, dataset):
        """Extract unique database schemas from the dataset"""
        db_schemas = {}
        
        # Process all examples to build schema information
        for split in ['train', 'validation']:
            for item in dataset[split]:
                db_id = item['db_id']
                if db_id not in db_schemas:
                    # Create a basic schema structure
                    db_schemas[db_id] = {
                        'db_id': db_id,
                        'table_names_original': [],
                        'table_names': [],
                        'column_names_original': [[-1, '*']],
                        'column_names': [[-1, '*']],
                        'column_types': ['text'],
                        'foreign_keys': [],
                        'primary_keys': []
                    }
        
        # Convert to list format
        return list(db_schemas.values())
    
    def load_spider_from_local(self):
        """Load Spider dataset from local files"""
        print("Loading Spider dataset from local files...")
        
        # Define file paths
        json_dir = self.data_dir / "json"
        train_file = json_dir / "train.json"
        dev_file = json_dir / "dev.json"
        tables_file = json_dir / "tables.json"
        
        # Alternative file names to check
        alt_train_file = json_dir / "train_spider.json"
        alt_dev_file = json_dir / "validation.json"
        
        # Load training data
        train_path = train_file if train_file.exists() else alt_train_file
        if train_path.exists():
            try:
                with open(train_path, 'r', encoding='utf-8') as f:
                    self.train_data = json.load(f)
                print(f"Loaded {len(self.train_data)} training examples from {train_path.name}")
            except Exception as e:
                print(f"Error loading training data: {e}")
                self.train_data = []
        else:
            print("Warning: No training data file found")
            self.train_data = []
        
        # Load development/validation data
        dev_path = dev_file if dev_file.exists() else alt_dev_file
        if dev_path.exists():
            try:
                with open(dev_path, 'r', encoding='utf-8') as f:
                    self.dev_data = json.load(f)
                print(f"Loaded {len(self.dev_data)} dev examples from {dev_path.name}")
            except Exception as e:
                print(f"Error loading dev data: {e}")
                self.dev_data = []
        else:
            print("Warning: No dev data file found")
            self.dev_data = []
        
        # Load tables/schema data
        if tables_file.exists():
            try:
                with open(tables_file, 'r', encoding='utf-8') as f:
                    self.tables_data = json.load(f)
                print(f"Loaded {len(self.tables_data)} database schemas from {tables_file.name}")
            except Exception as e:
                print(f"Error loading tables data: {e}")
                self.tables_data = self._create_basic_schemas()
        else:
            print("Warning: No tables.json found, creating basic schemas from data")
            self.tables_data = self._create_basic_schemas()
        
        # Validate loaded data
        if not self.train_data and not self.dev_data:
            print("No data found in local files. Using sample data.")
            return self._load_sample_data()
        
        print(f"Successfully loaded local Spider dataset:")
        print(f"  Training examples: {len(self.train_data)}")
        print(f"  Dev examples: {len(self.dev_data)}")
        print(f"  Database schemas: {len(self.tables_data)}")
        
        return True
    
    def _create_basic_schemas(self):
        """Create basic schema information from available data"""
        db_schemas = {}
        
        # Process all examples to build basic schema information
        all_data = (self.train_data or []) + (self.dev_data or [])
        
        for item in all_data:
            db_id = item.get('db_id')
            if db_id and db_id not in db_schemas:
                db_schemas[db_id] = {
                    'db_id': db_id,
                    'table_names_original': [],
                    'table_names': [],
                    'column_names_original': [[-1, '*']],
                    'column_names': [[-1, '*']],
                    'column_types': ['text'],
                    'foreign_keys': [],
                    'primary_keys': []
                }
        
        print(f"Created basic schemas for {len(db_schemas)} databases")
        return list(db_schemas.values())
    
    def _load_sample_data(self):
        """Load sample data as fallback"""
        print("Loading sample data as fallback...")
        
        sample_train_data = [
            {
                "db_id": "restaurant_1",
                "question": "How many restaurants are there?",
                "query": "SELECT count(*) FROM restaurant"
            },
            {
                "db_id": "restaurant_1", 
                "question": "What are the names of all restaurants?",
                "query": "SELECT name FROM restaurant"
            },
            {
                "db_id": "customer_1",
                "question": "Show me all customers",
                "query": "SELECT * FROM customer"
            },
            {
                "db_id": "hotel_1",
                "question": "List all hotels",
                "query": "SELECT * FROM hotel"
            },
            {
                "db_id": "booking_1",
                "question": "Show booking details",
                "query": "SELECT * FROM booking"
            }
        ]
        
        sample_tables_data = [
            {
                "db_id": "restaurant_1",
                "table_names_original": ["restaurant"],
                "table_names": ["restaurant"],
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "cuisine"]],
                "column_names": [[-1, "*"], [0, "id"], [0, "name"], [0, "cuisine"]],
                "column_types": ["text", "number", "text", "text"],
                "foreign_keys": [],
                "primary_keys": [1]
            },
            {
                "db_id": "customer_1",
                "table_names_original": ["customer"],
                "table_names": ["customer"],  
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "email"]],
                "column_names": [[-1, "*"], [0, "id"], [0, "name"], [0, "email"]],
                "column_types": ["text", "number", "text", "text"],
                "foreign_keys": [],
                "primary_keys": [1]
            },
            {
                "db_id": "hotel_1",
                "table_names_original": ["hotel"],
                "table_names": ["hotel"],
                "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "location"]],
                "column_names": [[-1, "*"], [0, "id"], [0, "name"], [0, "location"]],
                "column_types": ["text", "number", "text", "text"],
                "foreign_keys": [],
                "primary_keys": [1]
            }
        ]
        
        self.train_data = sample_train_data
        self.dev_data = sample_train_data[:2]
        self.tables_data = sample_tables_data
        
        print(f"Loaded {len(self.train_data)} sample training examples")
        print(f"Loaded {len(self.dev_data)} sample dev examples")
        print(f"Loaded {len(self.tables_data)} sample database schemas")
        print("Note: Using sample data. Please ensure Spider dataset is saved in ./spider_data/json/")
        
        return True
    
    def format_schema_for_embedding(self, table_info):
        """Format database schema information for embedding"""
        db_id = table_info['db_id']
        table_names = table_info.get('table_names_original', table_info.get('table_names', []))
        column_names = table_info.get('column_names_original', table_info.get('column_names', []))
        column_types = table_info.get('column_types', [])
        foreign_keys = table_info.get('foreign_keys', [])
        primary_keys = table_info.get('primary_keys', [])
        
        # Build schema text
        schema_parts = [f"Database: {db_id}"]
        
        # Add tables and columns
        for i, table_name in enumerate(table_names):
            schema_parts.append(f"Table: {table_name}")
            
            # Get columns for this table
            table_columns = []
            for j, col in enumerate(column_names):
                if isinstance(col, list) and len(col) >= 2 and col[0] == i:
                    col_type = column_types[j] if j < len(column_types) else 'unknown'
                    table_columns.append((j, col[1], col_type))
            
            for col_idx, col_name, col_type in table_columns:
                col_desc = f"  Column: {col_name} ({col_type})"
                if col_idx in primary_keys:
                    col_desc += " [PRIMARY KEY]"
                schema_parts.append(col_desc)
        
        # Add foreign key relationships
        if foreign_keys:
            schema_parts.append("Foreign Keys:")
            for fk in foreign_keys:
                if len(fk) >= 2:
                    try:
                        col1_idx, col2_idx = fk[0], fk[1]
                        if (col1_idx < len(column_names) and col2_idx < len(column_names)):
                            col1_info = column_names[col1_idx]
                            col2_info = column_names[col2_idx]
                            if (isinstance(col1_info, list) and isinstance(col2_info, list) and
                                len(col1_info) >= 2 and len(col2_info) >= 2 and
                                col1_info[0] < len(table_names) and col2_info[0] < len(table_names)):
                                table1 = table_names[col1_info[0]]
                                table2 = table_names[col2_info[0]]
                                schema_parts.append(f"  {table1}.{col1_info[1]} -> {table2}.{col2_info[1]}")
                    except (IndexError, TypeError):
                        continue
        
        return "\n".join(schema_parts)
    
    def format_question_for_embedding(self, example):
        """Format question and context for embedding"""
        question = example.get('question', '')
        db_id = example.get('db_id', 'unknown')
        
        return f"Database: {db_id}\nQuestion: {question}"
    
    def setup_collections(self):
        """Create ChromaDB collections"""
        # Delete existing collections if they exist
        for name in ["spider_schemas", "spider_questions", "spider_sql"]:
            try:
                self.client.delete_collection(name)
                print(f"Deleted existing collection: {name}")
            except:
                pass
        
        # Create new collections
        self.schema_collection = self.client.get_or_create_collection("spider_schemas")
        self.question_collection = self.client.get_or_create_collection("spider_questions")
        self.sql_collection = self.client.get_or_create_collection("spider_sql")
        
        print("Created ChromaDB collections")
    
    def store_schemas(self):
        """Store database schemas in ChromaDB"""
        if not self.tables_data:
            print("No schema data to store")
            return
            
        print("Storing database schemas...")
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for table_info in self.tables_data:
            schema_text = self.format_schema_for_embedding(table_info)
            embedding = self.model.encode(schema_text)
            
            documents.append(schema_text)
            embeddings.append(embedding.tolist())
            ids.append(table_info['db_id'])
            metadatas.append({
                "db_id": table_info['db_id'],
                "num_tables": len(table_info.get('table_names', [])),
                "num_columns": len(table_info.get('column_names', [])),
                "type": "schema"
            })
        
        # Add in batches to avoid size limits
        self._add_in_batches(
            self.schema_collection,
            documents, embeddings, ids, metadatas,
            batch_size=500  # Smaller batch size for schemas
        )
        
        print(f"Stored {len(documents)} database schemas")
    
    def store_questions(self, examples, split="train"):
        """Store questions in ChromaDB"""
        if not examples:
            print(f"No {split} data to store")
            return
            
        print(f"Storing {split} questions...")
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for i, example in enumerate(examples):
            question_text = self.format_question_for_embedding(example)
            embedding = self.model.encode(question_text)
            
            documents.append(question_text)
            embeddings.append(embedding.tolist())
            ids.append(f"{split}_{i}")
            metadatas.append({
                "db_id": example.get('db_id', 'unknown'),
                "question": example.get('question', ''),
                "sql": example.get('query', ''),
                "split": split,
                "type": "question"
            })
        
        # Add in batches to avoid size limits
        self._add_in_batches(
            self.question_collection,
            documents, embeddings, ids, metadatas,
            batch_size=1000
        )
        
        print(f"Stored {len(documents)} questions from {split} split")
    
    def store_sql_queries(self, examples, split="train"):
        """Store SQL queries for retrieval"""
        if not examples:
            print(f"No {split} SQL data to store")
            return
            
        print(f"Storing {split} SQL queries...")
        
        documents = []
        embeddings = []
        ids = []
        metadatas = []
        
        for i, example in enumerate(examples):
            sql_query = example.get('query', '')
            if not sql_query:
                continue
                
            embedding = self.model.encode(sql_query)
            
            documents.append(sql_query)
            embeddings.append(embedding.tolist())
            ids.append(f"{split}_sql_{i}")
            metadatas.append({
                "db_id": example.get('db_id', 'unknown'),
                "question": example.get('question', ''),
                "sql": sql_query,
                "split": split,
                "type": "sql"
            })
        
        # Add in batches to avoid size limits
        self._add_in_batches(
            self.sql_collection,
            documents, embeddings, ids, metadatas,
            batch_size=1000
        )
        
        print(f"Stored {len(documents)} SQL queries from {split} split")

def main():
    print("Spider Dataset + ChromaDB Integration Setup")
    print("=" * 50)
    
    # Initialize the integration
    try:
        spider_db = SpiderChromaDBIntegration()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return None
    
    # Load data from local files first
    print("Loading Spider dataset from local files...")
    success = spider_db.load_spider_from_local()
    
    # If local loading failed and HF is available, try HF as backup
    if not success and HF_AVAILABLE:
        print("Local loading failed. Attempting to load from Hugging Face as backup...")
        success = spider_db.load_spider_from_huggingface()
    
    if not success:
        print("Failed to load dataset from all sources.")
        return None
    
    # Setup and store data
    print("\nSetting up ChromaDB collections...")
    spider_db.setup_collections()
    
    print("\nStoring data in ChromaDB...")
    spider_db.store_schemas()
    spider_db.store_questions(spider_db.train_data, "train")
    spider_db.store_questions(spider_db.dev_data, "dev")
    spider_db.store_sql_queries(spider_db.train_data, "train")
    spider_db.store_sql_queries(spider_db.dev_data, "dev")
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    
    # Show summary
    schema_count = spider_db.schema_collection.count()
    question_count = spider_db.question_collection.count()
    sql_count = spider_db.sql_collection.count()
    
    print(f"\nSummary:")
    print(f"  Schemas stored: {schema_count}")
    print(f"  Questions stored: {question_count}")
    print(f"  SQL queries stored: {sql_count}")
    
    print(f"\nYou can now run:")
    print(f"  python interactive_spider_query.py")
    
    return spider_db

if __name__ == "__main__":
    spider_integration = main()