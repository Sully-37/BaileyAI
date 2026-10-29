import asyncio
import logging
import time

from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized inference runtime manager."""

    def __init__(self):
        self.stt = STTService()
        self.llm = LLMService()
        self.tts = TTSService()

        self.loaded = False
        self.loading = False
        self.last_error = None

        self._load_task = None
        self._load_lock = asyncio.Lock()

    async def start_loading(self):
        """Start loading and warm-up without holding the HTTP request open."""
        async with self._load_lock:
            if self.loaded:
                logger.info("MODEL_LOAD skipped reason=already_loaded")
                return

            if self._load_task and not self._load_task.done():
                logger.info("MODEL_LOAD skipped reason=already_loading")
                return

            self.loading = True
            self.last_error = None
            logger.info("MODEL_LOAD started")
            self._load_task = asyncio.create_task(self._load_all())

    async def _load_all(self):
        """Load models, run real inference, then mark Bailey ready."""
        total_started_at = time.perf_counter()

        try:
            await self._load_component("stt", self.stt.load)
            await self._load_component("llm", self.llm.load)
            # TTSService.load() already performs its own synthesis warm-up.
            await self._load_component("tts", self.tts.load)

            await self._warmup_component("stt", self.stt.warmup)
            await self._warmup_component("llm", self.llm.warmup)

            self.loaded = True
            self.last_error = None

            logger.info(
                "MODEL_LOAD complete total_ms=%s",
                round((time.perf_counter() - total_started_at) * 1000),
            )
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            logger.exception("MODEL_LOAD failed error=%s", exc)
        finally:
            self.loading = False

    async def _load_component(self, name: str, loader) -> None:
        started_at = time.perf_counter()
        logger.info("MODEL_LOAD component=%s status=starting", name)
        await loader()
        logger.info(
            "MODEL_LOAD component=%s status=complete elapsed_ms=%s",
            name,
            round((time.perf_counter() - started_at) * 1000),
        )

    async def _warmup_component(self, name: str, warmer) -> None:
        started_at = time.perf_counter()
        logger.info("MODEL_WARMUP component=%s status=starting", name)
        await warmer()
        logger.info(
            "MODEL_WARMUP component=%s status=complete elapsed_ms=%s",
            name,
            round((time.perf_counter() - started_at) * 1000),
        )

    def status(self):
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "stt": self.stt.loaded,
            "llm": self.llm.loaded,
            "tts": self.tts.loaded,
            "last_error": self.last_error,
        }


model_manager = ModelManager()
