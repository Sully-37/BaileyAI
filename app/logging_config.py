import logging
import sys

from app.config import LOG_LEVEL


def configure_logging() -> None:
    """
    Configures application-wide console logging.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )