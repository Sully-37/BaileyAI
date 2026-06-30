"""
Chatterbox Turbo Text-To-Speech Service.

Responsible for:
- loading Chatterbox Turbo into GPU memory
- generating speech audio
- cloning the configured reference voice
"""

import asyncio
import io
import logging

import soundfile as sf

from app.config import (
    CUDA_DEVICE,
    VOICE_REFERENCE_PATH,
)
from app.utils.gpu import gpu_is_available

logger = logging.getLogger(__name__)


class TTSService:
    """
    GPU resident TTS runtime.

    Chatterbox is imported lazily inside load() so the API process
    can start on CPU-only deploy-test servers.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Loads Chatterbox Turbo into GPU VRAM.
        """

        if not gpu_is_available():
            raise RuntimeError(
                "GPU unavailable. Chatterbox Turbo requires CUDA before model loading."
            )

        def _load():
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(
                device=CUDA_DEVICE,
            )

        try:
            self.model = await asyncio.to_thread(_load)
            self.loaded = True

        except Exception:
            logger.exception("Failed to load Chatterbox Turbo TTS runtime.")
            self.loaded = False
            raise

    async def synthesize(self, text: str) -> bytes:
        """
        Generates WAV audio bytes from text.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("TTS model is not loaded")

        def _generate():
            wav = self.model.generate(
                text=text,
                audio_prompt_path=VOICE_REFERENCE_PATH,
            )

            buffer = io.BytesIO()

            sf.write(
                buffer,
                wav.cpu().numpy(),
                self.model.sr,
                format="WAV",
            )

            buffer.seek(0)
            return buffer.read()

        return await asyncio.to_thread(_generate)