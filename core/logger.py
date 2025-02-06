import logging
import sys
from typing import Optional

def setup_logger(
    name: str,
    level: Optional[str] = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure logger with console-only output.
    
    Args:
        name: Name of the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    
    Returns:
        Configured logging.Logger instance
    
    Raises:
        ValueError: If invalid logging level is provided
    """
    # Validate logging level
    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        raise ValueError(f"Invalid logging level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Default format if none provided
    if not format_string:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

