"""Logging configuration utilities"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get configured logger
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
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