import asyncio
import tempfile

from faster_whisper import WhisperModel

from app.config import (
    CUDA_DEVICE,
    STT_MODEL_NAME,
    STT_COMPUTE_TYPE,
    STT_LANGUAGE,
)


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

        def _load():
            return WhisperModel(
                STT_MODEL_NAME,
                device=CUDA_DEVICE,
                compute_type=STT_COMPUTE_TYPE,
            )

        self.model = await asyncio.to_thread(_load)
        self.loaded = True

    async def transcribe_webm(self, chunks: list[bytes]) -> str:
        """
        Converts streamed browser WebM audio into text.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("STT model is not loaded")

        audio_bytes = b"".join(chunks)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as f:
            f.write(audio_bytes)
            f.flush()

            def _transcribe():
                segments, _ = self.model.transcribe(
                    f.name,
                    beam_size=1,
                    vad_filter=True,
                    language=STT_LANGUAGE,
                )

                return " ".join(seg.text.strip() for seg in segments)

            return await asyncio.to_thread(_transcribe)