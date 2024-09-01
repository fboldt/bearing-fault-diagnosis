import logging
import os
from datetime import datetime

def configure_logger(log_dir='experiments'):
    """Configures the logger."""
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{current_time}_experiment_log.txt"

    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ])

def log_message(message):
    if isinstance(message, dict):
        for key, value in message.items():
            logging.info(f"{key}: {value}")
    else:
        logging.info(message)
