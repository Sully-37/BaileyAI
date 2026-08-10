import logging
import sys

from app.config import LOG_LEVEL


def configure_logging() -> None:
    """
    Configures clear application-wide console logging.
    """

    logging.basicConfig(
        level=LOG_LEVEL,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Reduce dependency noise while keeping Bailey's timing logs visible.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)