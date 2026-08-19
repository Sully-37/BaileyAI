"""
CosyVoice 3 continuous bi-streaming Text-To-Speech Service.
"""

import asyncio
import io
import logging
import os
import queue
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

COSYVOICE_PROMPT_PREFIX = (
    "You are a helpful assistant.<|endofprompt|>"
)


class TTSService:
    """
    GPU-resident CosyVoice 3 runtime.

    One CosyVoice inference call remains alive for an entire
    Bailey response while Qwen feeds text chunks into it.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Downloads CosyVoice when needed, loads it, caches
        Bailey's voice, then performs one warm-up inference.
        """

        if self.loaded:
            return

        if not gpu_is_available():
            raise RuntimeError(
                "GPU unavailable. CosyVoice requires CUDA."
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
        Downloads CosyVoice only if it is not already cached.
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
            prompt_text = (
                COSYVOICE_PROMPT_PREFIX
                + VOICE_REFERENCE_TEXT
            )

            result = self.model.add_zero_shot_spk(
                prompt_text,
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
        Executes one discarded inference to warm CosyVoice.
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

    async def stream_audio_from_text_queue(
        self,
        text_queue: queue.Queue,
    ):
        """
        Runs one continuous CosyVoice inference.

        Qwen places short text chunks into text_queue.
        CosyVoice consumes those chunks through a generator
        while simultaneously producing streamed audio.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError(
                "TTS model is not loaded"
            )

        started_at = time.perf_counter()

        output_queue: asyncio.Queue = (
            asyncio.Queue()
        )

        loop = asyncio.get_running_loop()

        logger.info(
            "TTS_BISTREAM started"
        )

        def text_generator():
            """
            Bridges the thread-safe Qwen text queue into
            CosyVoice's synchronous generator interface.
            """

            text_index = 0

            while True:
                item = text_queue.get()

                if item is None:
                    logger.info(
                        "TTS_BISTREAM text_input_complete "
                        "chunks=%s",
                        text_index,
                    )
                    break

                if isinstance(item, Exception):
                    raise item

                text = str(item).strip()

                if not text:
                    continue

                text_index += 1

                logger.info(
                    "TTS_BISTREAM text_chunk index=%s "
                    "elapsed_ms=%s text=%r",
                    text_index,
                    round(
                        (
                            time.perf_counter()
                            - started_at
                        )
                        * 1000
                    ),
                    text,
                )

                yield text

        def _generate():
            try:
                for output in (
                    self.model.inference_zero_shot(
                        text_generator(),
                        "",
                        "",
                        zero_shot_spk_id=(
                            TTS_ZERO_SHOT_SPEAKER_ID
                        ),
                        stream=True,
                    )
                ):
                    waveform = output[
                        "tts_speech"
                    ]

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

                    audio_bytes = (
                        buffer.getvalue()
                    )

                    asyncio.run_coroutine_threadsafe(
                        output_queue.put(
                            audio_bytes
                        ),
                        loop,
                    ).result()

            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    output_queue.put(exc),
                    loop,
                ).result()

            finally:
                asyncio.run_coroutine_threadsafe(
                    output_queue.put(None),
                    loop,
                ).result()

        worker_task = asyncio.create_task(
            asyncio.to_thread(_generate)
        )

        audio_chunk_index = 0

        while True:
            item = await output_queue.get()

            if item is None:
                break

            if isinstance(item, Exception):
                raise item

            audio_chunk_index += 1

            logger.info(
                "TTS_BISTREAM audio_chunk index=%s "
                "elapsed_ms=%s bytes=%s",
                audio_chunk_index,
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

        await worker_task

        logger.info(
            "TTS_BISTREAM complete chunks=%s elapsed_ms=%s",
            audio_chunk_index,
            round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000
            ),
        )