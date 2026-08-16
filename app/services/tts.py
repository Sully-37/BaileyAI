"""
CosyVoice 3 streaming Text-To-Speech Service.
"""

import asyncio
import io
import logging
import os
import time

import numpy as np
import soundfile as sf

from huggingface_hub import snapshot_download

from app.config import (
    TTS_MODEL_NAME,
    TTS_MODEL_PATH,
    TTS_WARMUP_TEXT,
    TTS_ZERO_SHOT_SPEAKER_ID,
    VOICE_REFERENCE_PATH,
    VOICE_REFERENCE_TEXT,
)
from app.utils.gpu import gpu_is_available

logger = logging.getLogger(__name__)


class TTSService:
    """
    GPU-resident CosyVoice 3 streaming TTS runtime.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Downloads CosyVoice when needed, loads it, caches
        Bailey's voice, then performs a warm-up inference.
        """

        if self.loaded:
            return

        if not gpu_is_available():
            raise RuntimeError(
                "GPU unavailable. CosyVoice requires CUDA."
            )

        if VOICE_REFERENCE_TEXT.startswith(
            "REPLACE WITH"
        ):
            raise RuntimeError(
                "VOICE_REFERENCE_TEXT must contain the "
                "exact transcript of bailey_reference.wav."
            )

        load_started_at = time.perf_counter()

        try:
            await self._ensure_model_downloaded()

            def _load():
                from cosyvoice.cli.cosyvoice import AutoModel

                return AutoModel(
                    model_dir=TTS_MODEL_PATH,
                )

            self.model = await asyncio.to_thread(
                _load
            )

            logger.info(
                "TTS_LOAD model_ready elapsed_ms=%s",
                round(
                    (
                        time.perf_counter()
                        - load_started_at
                    )
                    * 1000
                ),
            )

            await self._cache_bailey_voice()
            await self._warmup()

            self.loaded = True

            logger.info(
                "TTS_LOAD complete total_ms=%s",
                round(
                    (
                        time.perf_counter()
                        - load_started_at
                    )
                    * 1000
                ),
            )

        except Exception:
            logger.exception("TTS_LOAD failed")
            self.loaded = False
            raise

    async def _ensure_model_downloaded(self):
        """
        Downloads CosyVoice 3 only when it is not already
        available on local disk.
        """

        if os.path.isdir(TTS_MODEL_PATH):
            logger.info(
                "TTS_MODEL cache_hit path=%s",
                TTS_MODEL_PATH,
            )
            return

        logger.info(
            "TTS_MODEL downloading model=%s",
            TTS_MODEL_NAME,
        )

        await asyncio.to_thread(
            snapshot_download,
            repo_id=TTS_MODEL_NAME,
            local_dir=TTS_MODEL_PATH,
        )

        logger.info(
            "TTS_MODEL download_complete path=%s",
            TTS_MODEL_PATH,
        )

    async def _cache_bailey_voice(self):
        """
        Builds Bailey's zero-shot speaker conditioning once.
        """

        logger.info(
            "TTS_VOICE_CACHE started speaker=%s",
            TTS_ZERO_SHOT_SPEAKER_ID,
        )

        started_at = time.perf_counter()

        def _cache():
            result = self.model.add_zero_shot_spk(
                VOICE_REFERENCE_TEXT,
                VOICE_REFERENCE_PATH,
                TTS_ZERO_SHOT_SPEAKER_ID,
            )

            if result is not True:
                raise RuntimeError(
                    "CosyVoice failed to cache Bailey's voice."
                )

        await asyncio.to_thread(_cache)

        logger.info(
            "TTS_VOICE_CACHE complete elapsed_ms=%s",
            round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000
            ),
        )

    async def _warmup(self):
        """
        Executes one discarded inference so the first user
        request does not pay the cold-start cost.
        """

        logger.info(
            "TTS_WARMUP started text=%r",
            TTS_WARMUP_TEXT,
        )

        started_at = time.perf_counter()

        def _run():
            for _ in self.model.inference_zero_shot(
                TTS_WARMUP_TEXT,
                "",
                "",
                zero_shot_spk_id=(
                    TTS_ZERO_SHOT_SPEAKER_ID
                ),
                stream=True,
            ):
                pass

        await asyncio.to_thread(_run)

        logger.info(
            "TTS_WARMUP complete elapsed_ms=%s",
            round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000
            ),
        )

    async def stream_audio(
        self,
        text: str,
    ):
        """
        Streams WAV chunks as CosyVoice produces them.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError(
                "TTS model is not loaded"
            )

        if not text.strip():
            raise ValueError(
                "TTS input text is empty"
            )

        logger.info(
            "TTS_STREAM started text_chars=%s text=%r",
            len(text),
            text,
        )

        started_at = time.perf_counter()

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _generate():
            try:
                for output in (
                    self.model.inference_zero_shot(
                        text,
                        "",
                        "",
                        zero_shot_spk_id=(
                            TTS_ZERO_SHOT_SPEAKER_ID
                        ),
                        stream=True,
                    )
                ):
                    waveform = output["tts_speech"]

                    audio_array = (
                        waveform
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                    )

                    audio_array = np.squeeze(
                        audio_array
                    )

                    buffer = io.BytesIO()

                    sf.write(
                        buffer,
                        audio_array,
                        self.model.sample_rate,
                        format="WAV",
                        subtype="PCM_16",
                    )

                    audio_bytes = buffer.getvalue()

                    asyncio.run_coroutine_threadsafe(
                        queue.put(audio_bytes),
                        loop,
                    ).result()

            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(exc),
                    loop,
                ).result()

            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(None),
                    loop,
                ).result()

        asyncio.create_task(
            asyncio.to_thread(_generate)
        )

        chunk_index = 0

        while True:
            item = await queue.get()

            if item is None:
                break

            if isinstance(item, Exception):
                raise item

            chunk_index += 1

            logger.info(
                "TTS_STREAM audio_chunk index=%s "
                "elapsed_ms=%s bytes=%s",
                chunk_index,
                round(
                    (
                        time.perf_counter()
                        - started_at
                    )
                    * 1000
                ),
                len(item),
            )

            yield item

        logger.info(
            "TTS_STREAM complete chunks=%s elapsed_ms=%s",
            chunk_index,
            round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000
            ),
        )