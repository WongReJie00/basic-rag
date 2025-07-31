import logging
import time
import functools

def setup_logging():
    """Configure logging for the RAG application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_app.log')
        ]
    )
    return logging.getLogger(__name__)

def timing_decorator(func):
    """decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"✅ {func.__name__}: {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {func.__name__}: {duration:.3f}s - {e}")
            raise
    return wrapper

def log_info(message):
    """Log info message"""
    logger = logging.getLogger(__name__)
    logger.info(message)

def log_error(message):
    """Log error message"""
    logger = logging.getLogger(__name__)
    logger.error(message)

