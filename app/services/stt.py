import asyncio
import tempfile

from faster_whisper import WhisperModel

from app.config import (
    CUDA_DEVICE,
    STT_COMPUTE_TYPE,
    STT_LANGUAGE,
    STT_MODEL_NAME,
)
from app.utils.gpu import gpu_is_available


class STTService:
    """
    Handles speech-to-text inference using Faster-Whisper.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Loads Whisper weights into GPU memory.
        """

        if self.loaded:
            return

        if not gpu_is_available():
            raise RuntimeError(
                "GPU unavailable. Faster-Whisper requires CUDA before model loading."
            )

        def _load():
            return WhisperModel(
                STT_MODEL_NAME,
                device=CUDA_DEVICE,
                compute_type=STT_COMPUTE_TYPE,
            )

        self.model = await asyncio.to_thread(_load)
        self.loaded = True

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".webm",
    ) -> str:
        """
        Transcribes an uploaded browser recording.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("STT model is not loaded")

        if not audio_bytes:
            raise ValueError("Audio payload is empty")

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=True,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()

            def _transcribe() -> str:
                segments, _ = self.model.transcribe(
                    temp_file.name,
                    beam_size=1,
                    vad_filter=True,
                    language=STT_LANGUAGE,
                )

                return " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.text.strip()
                ).strip()

            transcript = await asyncio.to_thread(_transcribe)

        if not transcript:
            raise RuntimeError("No speech was detected in the recording")

        return transcript

    async def transcribe_webm(self, chunks: list[bytes]) -> str:
        """
        Compatibility wrapper for the existing websocket implementation.
        """

        return await self.transcribe_bytes(
            audio_bytes=b"".join(chunks),
            suffix=".webm",
        )