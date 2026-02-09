"""WikiSQL dataset download and setup"""

from pathlib import Path
from typing import Dict, List, Optional
import json

from utils.file_io import read_json, write_json, ensure_directory
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class WikiSQLDatasetDownloader:
    """Download and setup WikiSQL dataset"""
    
    # Alternative dataset sources
    DATASET_SOURCES = [
        "wikisql",  # Original (deprecated)
        "wikitablequestions",  # Alternative
    ]
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wikisql_dir = self.data_dir / "wikisql"
        
    def download_from_huggingface(self) -> bool:
        """
        Download WikiSQL dataset from GitHub archive
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("📦 Downloading WikiSQL dataset from GitHub...")
        
        # Download the tar.bz2 archive directly from GitHub
        try:
            return self._download_tar_archive()
        except Exception as e:
            logger.error(f"❌ Error downloading from GitHub: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _download_tar_archive(self) -> bool:
        """
        Download WikiSQL tar.bz2 archive from GitHub
        
        Returns:
            True if successful
        """
        import urllib.request
        import tarfile
        import tempfile
        
        logger.info("Downloading data.tar.bz2 from GitHub...")
        
        # Direct link to the tar.bz2 file
        archive_url = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
        
        self.wikisql_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download to temporary file
            logger.info(f"Downloading from {archive_url}...")
            with tempfile.NamedTemporaryFile(suffix='.tar.bz2', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
                req = urllib.request.Request(archive_url)
                req.add_header('User-Agent', 'Mozilla/5.0')
                
                with urllib.request.urlopen(req, timeout=600) as response:
                    # Download with progress
                    file_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    block_size = 8192
                    
                    logger.info(f"File size: {file_size / (1024*1024):.1f} MB")
                    
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        
                        if file_size > 0:
                            progress = (downloaded / file_size) * 100
                            if downloaded % (block_size * 100) == 0:  # Log every ~800KB
                                logger.info(f"Downloaded: {progress:.1f}%")
            
            logger.info("✅ Download complete, extracting archive...")
            
            # Extract tar.bz2
            with tarfile.open(tmp_path, 'r:bz2') as tar:
                tar.extractall(path=self.data_dir)
            
            logger.info("✅ Extraction complete")
            
            # Clean up temp file
            import os
            os.unlink(tmp_path)
            
            # The archive extracts to a 'data' directory
            # Move files to wikisql directory
            extracted_data_dir = self.data_dir / "data"
            
            if extracted_data_dir.exists():
                logger.info("Moving files to wikisql directory...")
                
                # Move all files from data/ to wikisql/
                for item in extracted_data_dir.iterdir():
                    dest = self.wikisql_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            import shutil
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    item.rename(dest)
                
                # Remove empty data directory
                extracted_data_dir.rmdir()
                logger.info("✅ Files organized")
            
            # Convert JSONL files to JSON format
            logger.info("Converting JSONL to JSON format...")
            self._convert_jsonl_to_json()
            
            logger.info("✅ WikiSQL dataset downloaded successfully from GitHub!")
            return True
            
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error: {e.code} {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Failed to download archive: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _convert_jsonl_to_json(self) -> bool:
        """
        Convert JSONL files to JSON format and combine with table info
        
        Returns:
            True if successful
        """
        import json
        
        # Map of JSONL files to process
        splits = {
            'train': 'train.jsonl',
            'dev': 'dev.jsonl',
            'test': 'test.jsonl'
        }
        
        # Load all tables first
        all_tables = {}
        for split in ['train', 'dev', 'test']:
            tables_file = self.wikisql_dir / f"{split}.tables.jsonl"
            if tables_file.exists():
                logger.info(f"Loading tables from {tables_file.name}...")
                with open(tables_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            table = json.loads(line)
                            table_id = table.get('id')
                            if table_id and table_id not in all_tables:
                                all_tables[table_id] = table
        
        logger.info(f"Loaded {len(all_tables)} unique tables")
        
        # Process each split
        for split_name, jsonl_file in splits.items():
            input_path = self.wikisql_dir / jsonl_file
            output_path = self.wikisql_dir / f"{split_name}.json"
            
            if not input_path.exists():
                logger.warning(f"File not found: {input_path}")
                continue
            
            logger.info(f"Converting {jsonl_file}...")
            
            data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            
                            # Format the item
                            table_id = item.get('table_id', f"table_{idx}")
                            table_info = all_tables.get(table_id, {})
                            
                            formatted_item = {
                                'question_id': idx,
                                'db_id': table_id,
                                'question': item['question'],
                                'sql': self._format_sql_from_dict(item['sql']),
                                'query': self._format_sql_from_dict(item['sql']),
                                'table_id': table_id
                            }
                            
                            # Add table data if available
                            if table_info:
                                formatted_item['table'] = {
                                    'header': table_info.get('header', []),
                                    'rows': table_info.get('rows', [])[:5],
                                    'name': table_id,
                                    'types': table_info.get('types', [])
                                }
                            
                            data.append(formatted_item)
                            
                        except Exception as e:
                            logger.warning(f"Failed to parse line {idx} in {jsonl_file}: {e}")
                            continue
            
            # Save as JSON
            write_json(data, str(output_path))
            logger.info(f"✅ Saved {len(data)} examples to {split_name}.json")
        
        # Create tables.json with all unique tables
        logger.info("Creating tables.json...")
        formatted_tables = self._format_tables_from_dict(list(all_tables.values()))
        tables_file = self.wikisql_dir / "tables.json"
        write_json(formatted_tables, str(tables_file))
        logger.info(f"✅ Saved {len(formatted_tables)} tables")
        
        return True
    
    def _format_sql_from_dict(self, sql_dict: Dict) -> str:
        """Convert SQL dict to SQL string"""
        try:
            agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
            cond_ops = ['=', '>', '<', 'OP']
            
            sel = sql_dict.get('sel', 0)
            agg = sql_dict.get('agg', 0)
            conds = sql_dict.get('conds', [])
            
            # Build SELECT clause
            if agg == 0:
                select_clause = f"SELECT col{sel}"
            else:
                select_clause = f"SELECT {agg_ops[agg]}(col{sel})"
            
            # Build WHERE clause
            where_parts = []
            for cond in conds:
                if len(cond) >= 3:
                    col_idx, op_idx, value = cond[0], cond[1], cond[2]
                    op = cond_ops[op_idx] if op_idx < len(cond_ops) else '='
                    # Escape string values
                    if isinstance(value, str):
                        value = f"'{value}'"
                    where_parts.append(f"col{col_idx} {op} {value}")
            
            if where_parts:
                where_clause = " WHERE " + " AND ".join(where_parts)
            else:
                where_clause = ""
            
            return f"{select_clause} FROM table{where_clause}"
            
        except Exception as e:
            logger.warning(f"Failed to format SQL: {e}")
            return str(sql_dict)
    
    def _format_tables_from_dict(self, tables: List[Dict]) -> List[Dict]:
        """Format tables to Spider-compatible format"""
        formatted = []
        
        for table in tables:
            try:
                table_id = table.get('id', 'unknown')
                headers = table.get('header', [])
                types = table.get('types', ['text'] * len(headers))
                
                formatted_table = {
                    'db_id': table_id,
                    'table_names': [table_id],
                    'table_names_original': [table_id],
                    'column_names': [[0, col] for col in headers],
                    'column_names_original': [[0, col] for col in headers],
                    'column_types': types,
                    'primary_keys': [0] if headers else [],
                    'foreign_keys': [],
                    'rows': table.get('rows', [])[:5]  # Keep sample rows
                }
                formatted.append(formatted_table)
            except Exception as e:
                logger.warning(f"Failed to format table {table.get('id', 'unknown')}: {e}")
                continue
        
        return formatted
    
    def _format_huggingface_data(self, dataset) -> List[Dict]:
        """
        Format HuggingFace WikiSQL dataset to standardized format
        (Kept for backward compatibility but not used with tar archive)
        """
        data = []
        
        for idx, item in enumerate(dataset):
            try:
                if isinstance(item.get('sql'), dict):
                    sql_query = item['sql'].get('human_readable', '')
                    if not sql_query:
                        sql_query = self._format_sql_from_dict(item['sql'])
                else:
                    sql_query = str(item.get('sql', ''))
                
                table_id = item.get('table_id', f"table_{idx}")
                
                formatted_item = {
                    'question_id': idx,
                    'db_id': table_id,
                    'question': item['question'],
                    'sql': sql_query,
                    'query': sql_query,
                    'table': {
                        'header': item.get('table', {}).get('header', []),
                        'rows': item.get('table', {}).get('rows', [])[:5],
                        'name': table_id
                    }
                }
                
                data.append(formatted_item)
                
            except Exception as e:
                logger.warning(f"Failed to format item {idx}: {e}")
                continue
        
        return data
    
    def _extract_tables_info(self, dataset) -> List[Dict]:
        """
        Extract table schema information from WikiSQL dataset
        (Kept for backward compatibility but not used with tar archive)
        """
        tables_map = {}
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
                
            for idx, item in enumerate(dataset[split_name]):
                try:
                    table_id = item.get('table_id', f"table_{idx}")
                    
                    if table_id in tables_map:
                        continue
                    
                    table_data = item.get('table', {})
                    headers = table_data.get('header', [])
                    
                    columns = []
                    for col_idx, col_name in enumerate(headers):
                        col_type = "text"
                        rows = table_data.get('rows', [])
                        if rows and len(rows[0]) > col_idx:
                            first_val = rows[0][col_idx]
                            if isinstance(first_val, (int, float)):
                                col_type = "number"
                        
                        columns.append({
                            'column_name': col_name,
                            'column_type': col_type,
                            'is_primary_key': col_idx == 0
                        })
                    
                    tables_map[table_id] = {
                        'db_id': table_id,
                        'table_names': [table_id],
                        'table_names_original': [table_id],
                        'column_names': [[0, col] for col in headers],
                        'column_names_original': [[0, col] for col in headers],
                        'column_types': [col['column_type'] for col in columns],
                        'primary_keys': [0],
                        'foreign_keys': []
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to extract table info for item {idx}: {e}")
                    continue
        
        return list(tables_map.values())
    
    def download_dataset(self) -> bool:
        """
        Download WikiSQL dataset
        
        Returns:
            True if successful, False otherwise
        """
        return self.download_from_huggingface()
    
    def verify_dataset(self) -> Dict:
        """
        Verify downloaded dataset and return statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.wikisql_dir.exists():
            logger.error("WikiSQL dataset directory not found")
            return {}
        
        stats = {
            "train_exists": (self.wikisql_dir / "train.json").exists(),
            "dev_exists": (self.wikisql_dir / "dev.json").exists(),
            "test_exists": (self.wikisql_dir / "test.json").exists(),
            "tables_exists": (self.wikisql_dir / "tables.json").exists()
        }
        
        # Count examples in each split
        for split_name, file_name in [
            ("train", "train.json"),
            ("dev", "dev.json"),
            ("test", "test.json")
        ]:
            file_path = self.wikisql_dir / file_name
            if file_path.exists():
                data = read_json(str(file_path))
                stats[f"{split_name}_count"] = len(data)
        
        # Count tables
        tables_file = self.wikisql_dir / "tables.json"
        if tables_file.exists():
            tables = read_json(str(tables_file))
            stats["tables_count"] = len(tables)
        
        return stats