"""File I/O utility functions"""

from pathlib import Path
from typing import List, Dict, Any
import json
import zipfile
import requests


def read_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Read JSON file and return data
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def write_json(data: List[Dict[str, Any]], filepath: str, indent: int = 2) -> None:
    """
    Write data to JSON file
    
    Args:
        data: Data to write
        filepath: Path to output JSON file
        indent: JSON indentation level
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def write_txt(lines: List[str], filepath: str) -> None:
    """
    Write lines to text file
    
    Args:
        lines: List of lines to write
        filepath: Path to output text file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def ensure_directory(filepath: str) -> None:
    """
    Ensure directory exists for given filepath
    
    Args:
        filepath: Path to file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

def count_lines(filepath: str) -> int:
    """
    Count number of lines in a file
    
    Args:
        filepath: Path to file
        
    Returns:
        Number of lines
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)
    
def download_file(url: str, destination: Path, chunk_size: int = 8192, timeout: int = 30) -> bool:
    """
    Download file from URL
    
    Args:
        url: URL to download from
        destination: Path to save file
        chunk_size: Size of chunks to download
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract ZIP file
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False