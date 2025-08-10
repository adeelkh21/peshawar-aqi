"""
Logging Configuration for AQI Prediction System
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(data_dir):
    """
    Set up logging configuration with both file and console handlers
    
    Args:
        data_dir (str): Directory where log files should be stored
    """
    # Create logs directory
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    log_file = os.path.join(log_dir, f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    loggers = {
        'data_collection': logging.getLogger('data_collection'),
        'weather_api': logging.getLogger('weather_api'),
        'pollution_api': logging.getLogger('pollution_api'),
        'data_processing': logging.getLogger('data_processing'),
        'validation': logging.getLogger('validation')
    }
    
    return loggers
