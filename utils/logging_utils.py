"""Logging configuration utilities"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = None) -> logging.Logger:
    """
    Get a configured logger
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (None = use parent level)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if parent already has them
    if not logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(message)s')  # Simple format
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Only set level if explicitly provided
    if level is not None:
        logger.setLevel(level)
    
    return logger


def setup_file_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger that writes to file
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger with file handler
    """
    logger = get_logger(name, level)
    
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger