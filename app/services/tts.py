"""
Chatterbox Turbo Text-To-Speech Service.
"""

import asyncio
import io
import logging

import numpy as np
import soundfile as sf

from app.config import CUDA_DEVICE, VOICE_REFERENCE_PATH
from app.utils.gpu import gpu_is_available

logger = logging.getLogger(__name__)


class TTSService:
    """
    GPU-resident TTS runtime.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Loads Chatterbox Turbo into GPU VRAM.
        """

        if self.loaded:
            return

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

        if not text.strip():
            raise ValueError("TTS input text is empty")

        def _generate() -> bytes:
            waveform = self.model.generate(
                text=text,
                audio_prompt_path=VOICE_REFERENCE_PATH,
            )

            audio_array = waveform.detach().cpu().float().numpy()
            audio_array = np.squeeze(audio_array)

            buffer = io.BytesIO()

            sf.write(
                buffer,
                audio_array,
                self.model.sr,
                format="WAV",
                subtype="PCM_16",
            )

            buffer.seek(0)
            return buffer.read()

        return await asyncio.to_thread(_generate)