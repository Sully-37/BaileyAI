"""
Chatterbox Turbo Text-To-Speech Service.

Responsible for:
- loading Chatterbox Turbo into GPU memory
- generating speech audio
- cloning the configured reference voice
"""

import asyncio
import io

import soundfile as sf
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS

from app.config import (
    CUDA_DEVICE,
    VOICE_REFERENCE_PATH,
)


class TTSService:
    """
    GPU resident TTS runtime.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Loads Chatterbox Turbo into GPU VRAM.
        """

        def _load():
            return ChatterboxTurboTTS.from_pretrained(
                device=CUDA_DEVICE
            )

        self.model = await asyncio.to_thread(_load)

        self.loaded = True

    async def synthesize(self, text: str) -> bytes:
        """
        Generates WAV audio bytes from text.
        """

        if not self.loaded:
            raise RuntimeError("TTS model not loaded")

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