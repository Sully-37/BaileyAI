"""
Chatterbox Turbo Text-To-Speech Service.
"""

import asyncio
import io
import logging
import time

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
                "GPU unavailable. Chatterbox Turbo requires CUDA."
            )

        def _load():
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(
                device=CUDA_DEVICE,
            )

        started_at = time.perf_counter()

        try:
            self.model = await asyncio.to_thread(_load)
            self.loaded = True

            logger.info(
                "TTS_LOAD complete elapsed_ms=%s",
                round(
                    (time.perf_counter() - started_at) * 1000
                ),
            )

        except Exception:
            logger.exception(
                "TTS_LOAD failed"
            )

            self.loaded = False
            raise

    async def synthesize(self, text: str) -> bytes:
        """
        Generates WAV audio bytes.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("TTS model is not loaded")

        if not text.strip():
            raise ValueError("TTS input text is empty")

        logger.info(
            "TTS_INFERENCE started text_chars=%s",
            len(text),
        )

        started_at = time.perf_counter()

        def _generate() -> bytes:
            waveform = self.model.generate(
                text=text,
                audio_prompt_path=VOICE_REFERENCE_PATH,
            )

            audio_array = (
                waveform
                .detach()
                .cpu()
                .float()
                .numpy()
            )

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

        audio_bytes = await asyncio.to_thread(_generate)

        logger.info(
            "TTS_INFERENCE complete elapsed_ms=%s audio_bytes=%s",
            round((time.perf_counter() - started_at) * 1000),
            len(audio_bytes),
        )

        return audio_bytes