import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import re

class SpiderDatasetDownloader:
    """
    Enhanced Spider dataset handler that extracts table info from SQL queries
    when full schema isn't available.
    """
    
    def __init__(self, data_dir: str = "./spider_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def extract_tables_from_sql(self, sql_queries: List[str], db_ids: List[str]) -> Dict:
        """
        Extract table and column information from SQL queries.
        This creates a basic schema when full database files aren't available.
        """
        tables_info = {}
        
        for sql, db_id in zip(sql_queries, db_ids):
            if db_id not in tables_info:
                tables_info[db_id] = {
                    "db_id": db_id,
                    "table_names": set(),
                    "table_names_original": set(),
                    "column_names": set(),
                    "column_names_original": set(),
                    "column_types": [],
                    "foreign_keys": [],
                    "primary_keys": []
                }
            
            # Extract table names (basic regex - not perfect but functional)
            # Look for FROM, JOIN patterns
            from_matches = re.findall(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
            join_matches = re.findall(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
            
            for match in from_matches + join_matches:
                tables_info[db_id]["table_names"].add(match.lower())
                tables_info[db_id]["table_names_original"].add(match)
            
            # Extract column names (basic - looks for SELECT and WHERE clauses)
            # This is a simplified extraction
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_part = select_match.group(1)
                # Simple column extraction (doesn't handle all cases)
                columns = [col.strip() for col in select_part.split(',')]
                for col in columns:
                    if '.' in col:
                        table_col = col.split('.')[-1]
                        tables_info[db_id]["column_names"].add(table_col.lower())
                        tables_info[db_id]["column_names_original"].add(table_col)
        
        # Convert sets to lists for JSON serialization
        for db_id in tables_info:
            tables_info[db_id]["table_names"] = list(tables_info[db_id]["table_names"])
            tables_info[db_id]["table_names_original"] = list(tables_info[db_id]["table_names_original"])
            tables_info[db_id]["column_names"] = list(tables_info[db_id]["column_names"])
            tables_info[db_id]["column_names_original"] = list(tables_info[db_id]["column_names_original"])
        
        return tables_info
    
    def download_spider_dataset(self) -> bool:
        """
        Download Spider dataset using HuggingFace datasets library.
        Enhanced to create basic table information from SQL queries.
        """
        try:
            from datasets import load_dataset
            
            print("📦 Downloading Spider dataset from HuggingFace...")
            dataset = load_dataset("xlangai/spider")
            
            # Create directory structure
            spider_dir = self.data_dir / "spider"
            spider_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect all SQL queries and db_ids for table extraction
            all_sql_queries = []
            all_db_ids = []
            
            # Save train data
            train_data = []
            for item in dataset['train']:
                train_data.append({
                    'db_id': item['db_id'],
                    'question': item['question'],
                    'sql': item['query'],
                    'query': item['query'],
                    'query_toks': item['query_toks'],
                    'query_toks_no_value': item['query_toks_no_value'],
                    'question_toks': item['question_toks']
                })
                all_sql_queries.append(item['query'])
                all_db_ids.append(item['db_id'])
            
            with open(spider_dir / 'train_spider.json', 'w') as f:
                json.dump(train_data, f, indent=2)
            
            # Save validation data as dev.json
            dev_data = []
            for item in dataset['validation']:
                dev_data.append({
                    'db_id': item['db_id'],
                    'question': item['question'],
                    'sql': item['query'],
                    'query': item['query'],
                    'query_toks': item['query_toks'],
                    'query_toks_no_value': item['query_toks_no_value'],
                    'question_toks': item['question_toks']
                })
                all_sql_queries.append(item['query'])
                all_db_ids.append(item['db_id'])
            
            with open(spider_dir / 'dev.json', 'w') as f:
                json.dump(dev_data, f, indent=2)
            
            # Extract table information from SQL queries
            print("🔍 Extracting table information from SQL queries...")
            tables_info = self.extract_tables_from_sql(all_sql_queries, all_db_ids)
            
            # Convert to Spider's expected format
            tables_list = []
            for db_id, info in tables_info.items():
                tables_list.append({
                    "db_id": db_id,
                    "table_names": info["table_names"],
                    "table_names_original": info["table_names_original"],
                    "column_names": [[0, "*"]] + [[i, col] for i, col in enumerate(info["column_names"], 1)],
                    "column_names_original": [["*"]] + [[col] for col in info["column_names_original"]],
                    "column_types": ["text"] * (len(info["column_names"]) + 1),  # Default to text
                    "foreign_keys": info["foreign_keys"],
                    "primary_keys": info["primary_keys"]
                })
            
            with open(spider_dir / 'tables.json', 'w') as f:
                json.dump(tables_list, f, indent=2)
            
            print(f"✅ Downloaded {len(train_data)} training examples")
            print(f"✅ Downloaded {len(dev_data)} validation examples")
            print(f"✅ Extracted table info for {len(tables_info)} databases")
            print(f"📁 Saved to: {spider_dir}")
            print("\n⚠️  Note: Database .sqlite files not included.")
            print("⚠️  Table info extracted from SQL queries (basic schema only).")
            
            return True
            
        except ImportError:
            print("❌ HuggingFace datasets library not installed.")
            print("💡 Install with: pip install datasets")
            return False
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    
    def download_full_spider_dataset(self) -> bool:
        """
        Attempt to download the full Spider dataset with database files.
        This method tries to get the complete dataset including SQLite files.
        """
        try:
            import requests
            import zipfile
            
            print("🕷️ Attempting to download full Spider dataset...")
            
            # Try multiple sources for the full dataset
            urls = [
                "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ",
                "https://yale-lily.github.io/spider/spider.zip"
            ]
            
            for url in urls:
                try:
                    print(f"Trying {url}...")
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200:
                        zip_path = self.data_dir / "spider_full.zip"
                        with open(zip_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Try to extract
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(self.data_dir)
                        
                        zip_path.unlink()  # Remove zip file
                        print("✅ Full Spider dataset downloaded successfully!")
                        return True
                        
                except Exception as e:
                    print(f"Failed with {url}: {e}")
                    continue
            
            print("❌ Could not download full dataset. Using HuggingFace version.")
            return False
            
        except Exception as e:
            print(f"❌ Error downloading full dataset: {e}")
            return False
    
    def load_json_data(self, file_path: Path) -> List[Dict]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def explore_dataset(self) -> Dict:
        """Explore the downloaded Spider dataset structure."""
        spider_dir = self.data_dir / "spider"
        
        if not spider_dir.exists():
            print("Spider dataset not found. Please download first.")
            return {}
        
        # Load training and dev data
        train_data = self.load_json_data(spider_dir / "train_spider.json")
        dev_data = self.load_json_data(spider_dir / "dev.json")
        tables_data = self.load_json_data(spider_dir / "tables.json")
        
        # Check for database directory
        db_dir = spider_dir / "database"
        databases = []
        if db_dir.exists():
            databases = [d.name for d in db_dir.iterdir() if d.is_dir()]
        
        # Get unique database IDs from the data
        unique_db_ids = set()
        for item in train_data + dev_data:
            unique_db_ids.add(item['db_id'])
        
        dataset_info = {
            "train_examples": len(train_data),
            "dev_examples": len(dev_data),
            "tables": len(tables_data),
            "databases": len(databases) if databases else len(unique_db_ids),
            "database_names": list(databases)[:10] if databases else list(unique_db_ids)[:10],
            "unique_db_ids": list(unique_db_ids)
        }
        
        print("=== Spider Dataset Overview ===")
        print(f"Training examples: {dataset_info['train_examples']}")
        print(f"Development examples: {dataset_info['dev_examples']}")
        print(f"Number of table schemas: {dataset_info['tables']}")
        print(f"Number of unique databases: {len(unique_db_ids)}")
        print(f"Physical database files: {len(databases)}")
        
        if databases:
            print(f"Sample database files: {', '.join(dataset_info['database_names'])}")
        else:
            print("No physical database files (using extracted schema from SQL)")
            print(f"Database IDs: {', '.join(list(unique_db_ids)[:5])}{'...' if len(unique_db_ids) > 5 else ''}")
        
        return dataset_info
    
    def get_sample_data(self, num_samples: int = 5) -> Dict:
        """Get sample data from the dataset."""
        spider_dir = self.data_dir / "spider"
        
        if not spider_dir.exists():
            print("Spider dataset not found. Please download first.")
            return {}
        
        train_data = self.load_json_data(spider_dir / "train_spider.json")
        
        if not train_data:
            return {}
        
        samples = train_data[:num_samples]
        
        print("=== Sample Training Examples ===")
        for i, sample in enumerate(samples, 1):
            print(f"\nExample {i}:")
            print(f"Question: {sample['question']}")
            print(f"SQL: {sample.get('sql', sample.get('query', 'N/A'))}")
            print(f"Database: {sample['db_id']}")
            print("-" * 50)
        
        return {"samples": samples}
    
    def get_database_schema_info(self, db_id: str) -> Dict:
        """Get schema information for a specific database."""
        spider_dir = self.data_dir / "spider"
        tables_data = self.load_json_data(spider_dir / "tables.json")
        
        for table_info in tables_data:
            if table_info.get("db_id") == db_id:
                return table_info
        
        return {}
    
    def connect_to_database(self, db_name: str) -> Optional[sqlite3.Connection]:
        """Connect to a specific database in the dataset."""
        db_path = self.data_dir / "spider" / "database" / db_name / f"{db_name}.sqlite"
        
        if not db_path.exists():
            print(f"Database {db_name} not found at {db_path}")
            print("💡 Physical database files need the full Spider dataset download.")
            return None
        
        try:
            conn = sqlite3.connect(str(db_path))
            print(f"Connected to database: {db_name}")
            return conn
        except Exception as e:
            print(f"Error connecting to database {db_name}: {e}")
            return None
    
    def explore_database_schema(self, db_name: str) -> Dict:
        """Explore the schema of a specific database."""
        # First try to connect to physical database
        conn = self.connect_to_database(db_name)
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                schema_info = {"database": db_name, "tables": {}}
                
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = cursor.fetchall()
                    schema_info["tables"][table] = [
                        {"name": col[1], "type": col[2], "nullable": not col[3]} 
                        for col in columns
                    ]
                
                print(f"=== Schema for {db_name} (from SQLite) ===")
                for table, columns in schema_info["tables"].items():
                    print(f"\nTable: {table}")
                    for col in columns:
                        print(f"  - {col['name']} ({col['type']})")
                
                conn.close()
                return schema_info
                
            except Exception as e:
                print(f"Error exploring schema: {e}")
                conn.close()
                return {}
        
        # Fallback to extracted schema info
        schema_info = self.get_database_schema_info(db_name)
        if schema_info:
            print(f"=== Schema for {db_name} (extracted from SQL) ===")
            print(f"Tables: {', '.join(schema_info.get('table_names', []))}")
            print(f"Columns: {len(schema_info.get('column_names', []))} columns found")
        else:
            print(f"No schema information found for {db_name}")
        
        return schema_info
    
    def create_dataframe(self, split: str = "train") -> pd.DataFrame:
        """Create a pandas DataFrame from the dataset."""
        spider_dir = self.data_dir / "spider"
        
        if split == "train":
            data = self.load_json_data(spider_dir / "train_spider.json")
        elif split == "dev":
            data = self.load_json_data(spider_dir / "dev.json")
        else:
            print("Split must be 'train' or 'dev'")
            return pd.DataFrame()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        return df

# Usage example
def main():
    """Enhanced main function with better dataset handling."""
    downloader = SpiderDatasetDownloader("./spider_dataset")
    
    print("🕷️ Spider Dataset Downloader")
    print("=" * 50)
    
    # Try full dataset first, then fallback to HuggingFace
    print("Step 1: Downloading Spider dataset...")
    if not downloader.download_full_spider_dataset():
        print("Falling back to HuggingFace version...")
        if not downloader.download_spider_dataset():
            print("❌ All download methods failed")
            return
    
    print("\nStep 2: Exploring dataset...")
    dataset_info = downloader.explore_dataset()
    
    print("\nStep 3: Sample data...")
    downloader.get_sample_data(num_samples=3)
    
    print("\nStep 4: Schema exploration...")
    if dataset_info.get("database_names"):
        sample_db = dataset_info["database_names"][0]
        downloader.explore_database_schema(sample_db)
    
    print("\nStep 5: Creating DataFrames...")
    train_df = downloader.create_dataframe("train")
    dev_df = downloader.create_dataframe("dev")
    
    if not train_df.empty:
        print(f"✅ Training data: {train_df.shape}")
        print(f"✅ Dev data: {dev_df.shape}")

if __name__ == "__main__":
    main()