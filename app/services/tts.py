"""
Pocket TTS streaming Text-To-Speech service.
"""

import asyncio
import logging
import queue
import time

import numpy as np

from pocket_tts import TTSModel

from app.config import (
    TTS_WARMUP_TEXT,
    VOICE_REFERENCE_PATH,
)

logger = logging.getLogger(__name__)


class TTSService:
    """
    CPU-resident Pocket TTS runtime.

    Bailey's voice state is created once during startup and
    reused for every synthesis request.
    """

    def __init__(self):
        self.model = None
        self.voice_state = None
        self.loaded = False
        self.sample_rate = None

    async def load(self):
        """
        Loads Pocket TTS and prepares Bailey's cloned voice.
        """

        if self.loaded:
            return

        started_at = time.perf_counter()

        try:
            self.model = await asyncio.to_thread(
                TTSModel.load_model
            )

            self.sample_rate = self.model.sample_rate

            logger.info(
                "TTS_LOAD model_ready elapsed_ms=%s device=%s",
                round(
                    (time.perf_counter() - started_at)
                    * 1000
                ),
                self.model.device,
            )

            voice_started_at = time.perf_counter()

            self.voice_state = await asyncio.to_thread(
                self.model.get_state_for_audio_prompt,
                VOICE_REFERENCE_PATH,
            )

            logger.info(
                "TTS_VOICE ready elapsed_ms=%s",
                round(
                    (
                        time.perf_counter()
                        - voice_started_at
                    )
                    * 1000
                ),
            )

            await self._warmup()

            self.loaded = True

            logger.info(
                "TTS_LOAD complete total_ms=%s sample_rate=%s",
                round(
                    (time.perf_counter() - started_at)
                    * 1000
                ),
                self.sample_rate,
            )

        except Exception:
            logger.exception("TTS_LOAD failed")
            self.loaded = False
            raise

    async def _warmup(self):
        """
        Runs one discarded streaming inference.
        """

        def _run():
            for _ in self.model.generate_audio_stream(
                self.voice_state,
                TTS_WARMUP_TEXT,
            ):
                pass

        started_at = time.perf_counter()

        await asyncio.to_thread(_run)

        logger.info(
            "TTS_WARMUP complete elapsed_ms=%s",
            round(
                (time.perf_counter() - started_at)
                * 1000
            ),
        )

    async def stream_audio_from_text_queue(
        self,
        text_queue: queue.Queue,
    ):
        """
        Streams raw PCM16 audio as Qwen text chunks arrive.

        Each Qwen text chunk is synthesized with Pocket's
        native streaming generator. Audio frames are yielded
        immediately instead of waiting for a complete WAV.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError(
                "TTS model is not loaded"
            )

        loop = asyncio.get_running_loop()
        output_queue = asyncio.Queue()

        started_at = time.perf_counter()

        def _generate():
            audio_index = 0

            try:
                while True:
                    item = text_queue.get()

                    if item is None:
                        break

                    if isinstance(item, Exception):
                        raise item

                    text = str(item).strip()

                    if not text:
                        continue

                    logger.info(
                        "TTS text_chunk text=%r",
                        text,
                    )

                    for audio in (
                        self.model.generate_audio_stream(
                            self.voice_state,
                            text,
                        )
                    ):
                        audio_array = (
                            audio
                            .detach()
                            .cpu()
                            .float()
                            .numpy()
                        )

                        audio_array = np.squeeze(
                            audio_array
                        )

                        audio_array = np.clip(
                            audio_array,
                            -1.0,
                            1.0,
                        )

                        pcm16 = (
                            audio_array * 32767.0
                        ).astype("<i2")

                        audio_bytes = pcm16.tobytes()

                        audio_index += 1

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

        worker = asyncio.create_task(
            asyncio.to_thread(_generate)
        )

        chunk_index = 0

        while True:
            item = await output_queue.get()

            if item is None:
                break

            if isinstance(item, Exception):
                raise item

            chunk_index += 1

            logger.info(
                "TTS_AUDIO chunk=%s elapsed_ms=%s bytes=%s",
                chunk_index,
                round(
                    (time.perf_counter() - started_at)
                    * 1000
                ),
                len(item),
            )

            yield item

        await worker