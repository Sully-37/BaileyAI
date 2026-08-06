import asyncio

from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService


class ModelManager:
    """
    Centralized GPU runtime manager.
    """

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
        """
        Starts model loading in the background without holding the HTTP request open.
        """

        async with self._load_lock:
            if self.loaded:
                return

            if self._load_task and not self._load_task.done():
                return

            self.loading = True
            self.last_error = None
            self._load_task = asyncio.create_task(self._load_all())

    async def _load_all(self):
        """
        Loads all inference runtimes into GPU VRAM.
        """

        try:
            await self.stt.load()
            await self.llm.load()
            await self.tts.load()

            self.loaded = True
            self.last_error = None

        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)

        finally:
            self.loading = False

    def status(self):
        """
        Returns the current model-loading state.
        """

        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "stt": self.stt.loaded,
            "llm": self.llm.loaded,
            "tts": self.tts.loaded,
            "last_error": self.last_error,
        }


model_manager = ModelManager()
