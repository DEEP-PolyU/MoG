import logging
import sys
from datetime import datetime
from typing import Optional

__all__ = ["logger", "setup_logger"]

COLORS = {
    'DEBUG': '\033[0;36m',    # Cyan
    'INFO': '\033[0;32m',     # Green
    'WARNING': '\033[0;33m',  # Yellow
    'ERROR': '\033[0;31m',    # Red
    'CRITICAL': '\033[0;35m', # Magenta
    'RESET': '\033[0m'        # Reset color
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        formatted = super().format(record)
        color = COLORS.get(record.levelname)
        if color:
            return f"{color}{formatted}{COLORS['RESET']}"
        return formatted

def setup_logger(name: str = "youtu-graphrag", 
                level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = ColoredFormatter(
        fmt='[%(asctime)s] %(levelname)-8s %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-8s %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# logger = setup_logger(level=logging.WARNING)
logger = setup_logger(level=logging.INFO)


# Usage example:
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
