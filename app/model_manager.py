from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService


class ModelManager:
    """
    Centralized GPU runtime manager.

    Responsible for:
    - loading models into VRAM
    - maintaining warm inference state
    - exposing singleton model services
    """

    def __init__(self):
        # Streaming speech-to-text runtime.
        self.stt = STTService()

        # Conversational language model runtime.
        self.llm = LLMService()

        # Text-to-speech runtime.
        self.tts = TTSService()

        # Indicates whether all models are loaded.
        self.loaded = False

        # Stores the latest model load error if startup fails.
        self.last_error = None

    async def load_all(self):
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
            raise

    def status(self):
        """
        Returns current model residency state.
        """
        return {
            "loaded": self.loaded,
            "stt": self.stt.loaded,
            "llm": self.llm.loaded,
            "tts": self.tts.loaded,
            "last_error": self.last_error,
        }


model_manager = ModelManager()