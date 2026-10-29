import logging
import sys

from app.config import LOG_LEVEL


def configure_logging() -> None:
    """Configure concise application logs while retaining warnings and errors."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Third-party models can log on every generated audio chunk.
    for name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "faster_whisper",
        "pocket_tts",
        "transformers",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Service-level details remain available at DEBUG. Request milestones
    # from app.api.bailey remain visible at INFO.
    for name in (
        "app.services.stt",
        "app.services.llm",
        "app.services.tts",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
