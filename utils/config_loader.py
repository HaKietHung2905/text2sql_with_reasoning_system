"""Configuration file loading utilities"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Parsed configuration
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON configuration file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed configuration
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration file (auto-detect format)
    
    Args:
        filepath: Path to config file
        
    Returns:
        Parsed configuration
    """
    path = Path(filepath)
    
    if path.suffix in ['.yaml', '.yml']:
        return load_yaml(filepath)
    elif path.suffix == '.json':
        return load_json(filepath)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")