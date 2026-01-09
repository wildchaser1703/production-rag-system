import sys
from loguru import logger
from src.rag_pipeline.config import settings

def setup_logger() -> None:
    """
    Configures loguru logger with standard production formatting.
    """
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
    )
    
    # Add file handler for production auditing
    logger.add(
        "logs/rag_pipeline.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        compression="zip",
        serialize=True # JSON format for easier parsing by log aggregators
    )

setup_logger()
log = logger
