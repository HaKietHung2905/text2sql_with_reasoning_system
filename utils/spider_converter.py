"""Spider dataset format conversion utilities"""

from typing import List, Dict, Any
from utils.file_io import read_json, write_txt, ensure_directory
from utils.text_utils import create_tab_separated_line
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpiderConverter:
    """Convert Spider dataset between different formats"""
    
    def __init__(self):
        pass
    
    def json_to_sql_txt(self, input_json_file: str, output_txt_file: str) -> int:
        """
        Convert Spider JSON to SQL queries in tab-separated format.
        Format: SQL_QUERY\tDATABASE_ID
        
        Args:
            input_json_file: Path to input JSON file
            output_txt_file: Path to output TXT file
            
        Returns:
            Number of entries converted
        """
        logger.info(f"Converting {input_json_file} to SQL format...")
        
        # Read JSON data
        data = read_json(input_json_file)
        
        # Ensure output directory exists
        ensure_directory(output_txt_file)
        
        # Convert to lines
        lines = []
        for item in data:
            sql_query = item.get('sql', item.get('query', ''))
            db_id = item.get('db_id', '')
            
            if sql_query and db_id:
                lines.append(create_tab_separated_line(sql_query, db_id))
            else:
                logger.warning(f"Skipping item with missing sql or db_id: {item}")
        
        # Write to file
        write_txt(lines, output_txt_file)
        
        logger.info(f"Converted {len(lines)} SQL queries to {output_txt_file}")
        return len(lines)
    
    def json_to_questions_txt(self, input_json_file: str, output_txt_file: str) -> int:
        """
        Convert Spider JSON to questions in tab-separated format.
        Format: QUESTION\tDATABASE_ID
        
        Args:
            input_json_file: Path to input JSON file
            output_txt_file: Path to output questions TXT file
            
        Returns:
            Number of questions converted
        """
        logger.info(f"Converting {input_json_file} to questions format...")
        
        # Read JSON data
        data = read_json(input_json_file)
        
        # Ensure output directory exists
        ensure_directory(output_txt_file)
        
        # Convert to lines
        lines = []
        for item in data:
            question = item.get('question', '')
            db_id = item.get('db_id', '')
            
            if question and db_id:
                lines.append(create_tab_separated_line(question, db_id))
            else:
                logger.warning(f"Skipping item with missing question or db_id: {item}")
        
        # Write to file
        write_txt(lines, output_txt_file)
        
        logger.info(f"Converted {len(lines)} questions to {output_txt_file}")
        return len(lines)
    
    def json_to_full_format(self, input_json_file: str, output_txt_file: str) -> int:
        """
        Convert Spider JSON to full format with question, SQL, and database.
        Format: QUESTION\tSQL_QUERY\tDATABASE_ID
        
        Args:
            input_json_file: Path to input JSON file
            output_txt_file: Path to output TXT file
            
        Returns:
            Number of entries converted
        """
        logger.info(f"Converting {input_json_file} to full format...")
        
        # Read JSON data
        data = read_json(input_json_file)
        
        # Ensure output directory exists
        ensure_directory(output_txt_file)
        
        # Convert to lines
        lines = []
        for item in data:
            question = item.get('question', '')
            sql_query = item.get('sql', item.get('query', ''))
            db_id = item.get('db_id', '')
            
            if question and sql_query and db_id:
                line = f"{question}\t{sql_query}\t{db_id}\n"
                lines.append(line)
            else:
                logger.warning(f"Skipping incomplete item: {item}")
        
        # Write to file
        write_txt(lines, output_txt_file)
        
        logger.info(f"Converted {len(lines)} entries to {output_txt_file}")
        return len(lines)


def convert_spider_dataset(
    json_file: str,
    output_dir: str,
    dataset_name: str = "dev"
) -> Dict[str, str]:
    """
    Convert Spider JSON dataset to all formats
    
    Args:
        json_file: Path to Spider JSON file
        output_dir: Output directory for converted files
        dataset_name: Name of dataset (e.g., 'dev', 'train')
        
    Returns:
        Dictionary with paths to generated files
    """
    converter = SpiderConverter()
    
    output_files = {
        'sql': f"{output_dir}/{dataset_name}_gold_queries.txt",
        'questions': f"{output_dir}/{dataset_name}_questions.txt",
        'full': f"{output_dir}/{dataset_name}_full.txt"
    }
    
    # Convert to all formats
    converter.json_to_sql_txt(json_file, output_files['sql'])
    converter.json_to_questions_txt(json_file, output_files['questions'])
    converter.json_to_full_format(json_file, output_files['full'])
    
    return output_files