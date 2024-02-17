import logging
import logging.config

from rich.logging import RichHandler


def setup_logger(log_level=logging.WARNING):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    logging.getLogger().setLevel(log_level)
    return logger

# Initialize logger
logger = setup_logger()
