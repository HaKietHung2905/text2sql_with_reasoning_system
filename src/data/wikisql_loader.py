"""WikiSQL dataset loading and processing"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from utils.file_io import read_json, write_json, ensure_directory
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class WikiSQLDataLoader:
    """Load and process WikiSQL dataset files"""
    
    def __init__(self, data_dir: str = "./data/raw/wikisql"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_split(self, split: str = "train") -> List[Dict]:
        """
        Load a specific split of the dataset
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            
        Returns:
            List of examples
        """
        if split == "train":
            filepath = self.data_dir / "train.json"
        elif split == "dev":
            filepath = self.data_dir / "dev.json"
        elif split == "test":
            filepath = self.data_dir / "test.json"
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []
        
        data = read_json(str(filepath))
        logger.info(f"Loaded {len(data)} examples from {split} split")
        return data
    
    def load_tables(self) -> List[Dict]:
        """
        Load table schema information
        
        Returns:
            List of table schemas
        """
        filepath = self.data_dir / "tables.json"
        
        if not filepath.exists():
            logger.warning(f"Tables file not found: {filepath}")
            return []
        
        tables = read_json(str(filepath))
        logger.info(f"Loaded {len(tables)} table schemas")
        return tables
    
    def get_table_schema(self, db_id: str) -> Optional[Dict]:
        """
        Get schema for a specific database/table
        
        Args:
            db_id: Database identifier
            
        Returns:
            Table schema dictionary or None
        """
        tables = self.load_tables()
        
        for table in tables:
            if table.get('db_id') == db_id:
                return table
        
        logger.warning(f"Schema not found for db_id: {db_id}")
        return None
    
    def get_samples(self, split: str = "train", n: int = 5) -> List[Dict]:
        """
        Get sample examples from a split
        
        Args:
            split: Dataset split
            n: Number of samples
            
        Returns:
            List of sample examples
        """
        data = self.load_split(split)
        
        if not data:
            return []
        
        return data[:min(n, len(data))]
    
    def get_stats(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        for split in ['train', 'dev', 'test']:
            data = self.load_split(split)
            stats[f'{split}_size'] = len(data)
            
            if data:
                # Count unique tables (in WikiSQL each question has its own table)
                unique_tables = len(set(item['db_id'] for item in data))
                stats[f'{split}_unique_tables'] = unique_tables
        
        tables = self.load_tables()
        stats['total_tables'] = len(tables)
        
        return stats
    
    def filter_by_difficulty(self, split: str = "dev") -> Dict[str, List[Dict]]:
        """
        Filter examples by difficulty (based on query complexity)
        WikiSQL queries are generally simple (single table), so we categorize by:
        - Easy: Simple SELECT with WHERE
        - Medium: Aggregation functions
        - Hard: Multiple conditions or complex aggregations
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary mapping difficulty levels to examples
        """
        data = self.load_split(split)
        
        difficulty_groups = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for item in data:
            sql = item.get('sql', '').upper()
            
            # Count complexity indicators
            has_aggregation = any(agg in sql for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN'])
            has_where = 'WHERE' in sql
            has_group_by = 'GROUP BY' in sql
            has_order_by = 'ORDER BY' in sql
            and_count = sql.count(' AND ')
            or_count = sql.count(' OR ')
            
            # Classify difficulty
            complexity_score = (
                (1 if has_aggregation else 0) +
                (1 if has_group_by else 0) +
                (1 if has_order_by else 0) +
                and_count +
                or_count
            )
            
            if complexity_score == 0:
                difficulty_groups['easy'].append(item)
            elif complexity_score <= 2:
                difficulty_groups['medium'].append(item)
            else:
                difficulty_groups['hard'].append(item)
        
        # Log distribution
        logger.info("Difficulty distribution:")
        for level, items in difficulty_groups.items():
            logger.info(f"  {level}: {len(items)} examples")
        
        return difficulty_groups
    
    def export_to_csv(self, split: str = "dev", output_path: str = None):
        """
        Export a split to CSV format
        
        Args:
            split: Dataset split to export
            output_path: Path for output CSV file
        """
        data = self.load_split(split)
        
        if not data:
            logger.warning(f"No data to export for split: {split}")
            return
        
        # Flatten data for CSV
        rows = []
        for item in data:
            rows.append({
                'question_id': item.get('question_id', ''),
                'db_id': item.get('db_id', ''),
                'question': item.get('question', ''),
                'sql': item.get('sql', ''),
                'table_name': item.get('table', {}).get('name', ''),
                'num_columns': len(item.get('table', {}).get('header', []))
            })
        
        df = pd.DataFrame(rows)
        
        if output_path is None:
            output_path = self.data_dir / f"{split}_export.csv"
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rows)} examples to {output_path}")


class WikiSQLSchemaExtractor:
    """Extract and format WikiSQL table schemas"""
    
    def __init__(self):
        pass
    
    def extract_from_item(self, item: Dict) -> Dict:
        """
        Extract schema from a WikiSQL data item
        
        Args:
            item: WikiSQL data item with table information
            
        Returns:
            Formatted schema dictionary
        """
        table = item.get('table', {})
        header = table.get('header', [])
        
        schema = {
            'db_id': item.get('db_id', 'unknown'),
            'table_name': table.get('name', 'wikisql_data'),
            'columns': []
        }
        
        for idx, col_name in enumerate(header):
            schema['columns'].append({
                'name': col_name,
                'type': 'text',  # WikiSQL doesn't specify types
                'is_primary': idx == 0
            })
        
        return schema
    
    def format_schema_string(self, item: Dict) -> str:
        """
        Format schema as a string for prompt context
        
        Args:
            item: WikiSQL data item
            
        Returns:
            Formatted schema string
        """
        schema = self.extract_from_item(item)
        table_name = schema['table_name']
        columns = ', '.join([col['name'] for col in schema['columns']])
        
        return f"Table: {table_name}\nColumns: {columns}"
    
    def get_table_context(self, item: Dict, include_sample_data: bool = True) -> str:
        """
        Get full table context including schema and sample data
        
        Args:
            item: WikiSQL data item
            include_sample_data: Whether to include sample rows
            
        Returns:
            Formatted context string
        """
        context = self.format_schema_string(item)
        
        if include_sample_data and 'table' in item:
            table = item['table']
            if 'rows' in table and table['rows']:
                context += "\n\nSample data:"
                for i, row in enumerate(table['rows'][:3], 1):
                    context += f"\n  Row {i}: {row}"
        
        return context