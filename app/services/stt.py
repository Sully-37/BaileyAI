import asyncio
import logging
import tempfile
import time

from faster_whisper import WhisperModel

from app.config import (
    CUDA_DEVICE,
    STT_COMPUTE_TYPE,
    STT_LANGUAGE,
    STT_MODEL_NAME,
    STT_VAD_MIN_SILENCE_MS,
)
from app.utils.gpu import gpu_is_available

logger = logging.getLogger(__name__)


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
                "GPU unavailable. Faster-Whisper requires CUDA."
            )

        def _load():
            return WhisperModel(
                STT_MODEL_NAME,
                device=CUDA_DEVICE,
                compute_type=STT_COMPUTE_TYPE,
            )

        started_at = time.perf_counter()

        self.model = await asyncio.to_thread(_load)
        self.loaded = True

        logger.info(
            "STT_LOAD complete model=%s elapsed_ms=%s",
            STT_MODEL_NAME,
            round((time.perf_counter() - started_at) * 1000),
        )

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".webm",
    ) -> str:
        """
        Transcribes uploaded browser audio.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("STT model is not loaded")

        if not audio_bytes:
            raise ValueError("Audio payload is empty")

        logger.info(
            "STT_INFERENCE started audio_bytes=%s",
            len(audio_bytes),
        )

        started_at = time.perf_counter()

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
                    vad_parameters={
                        "min_silence_duration_ms": (
                            STT_VAD_MIN_SILENCE_MS
                        ),
                    },
                    language=STT_LANGUAGE,
                )

                return " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.text.strip()
                ).strip()

            transcript = await asyncio.to_thread(_transcribe)

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000
        )

        if not transcript:
            logger.warning(
                "STT_INFERENCE complete elapsed_ms=%s result=no_speech",
                elapsed_ms,
            )

            raise RuntimeError(
                "No speech was detected in the recording"
            )

        logger.info(
            "STT_INFERENCE complete elapsed_ms=%s "
            "transcript_chars=%s transcript=%r",
            elapsed_ms,
            len(transcript),
            transcript,
        )

        return transcript

    async def transcribe_webm(
        self,
        chunks: list[bytes],
    ) -> str:
        """
        Compatibility wrapper for websocket audio.
        """

        return await self.transcribe_bytes(
            audio_bytes=b"".join(chunks),
            suffix=".webm",
        )