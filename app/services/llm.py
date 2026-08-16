import asyncio
import logging
import re
import threading
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from app.config import (
    CUDA_DEVICE,
    LLM_CHUNK_MAX_WORDS,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)
from app.utils.gpu import gpu_is_available

logger = logging.getLogger(__name__)


class LLMService:
    """
    GPU-resident conversational language model runtime.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    async def load(self):
        """
        Loads Qwen into GPU memory.
        """

        if self.loaded:
            return

        if not gpu_is_available():
            raise RuntimeError(
                "GPU unavailable. Qwen inference requires CUDA."
            )

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            model.eval()

            return tokenizer, model

        started_at = time.perf_counter()

        self.tokenizer, self.model = await asyncio.to_thread(
            _load
        )

        self.loaded = True

        logger.info(
            "LLM_LOAD complete model=%s elapsed_ms=%s",
            LLM_MODEL_NAME,
            round(
                (time.perf_counter() - started_at)
                * 1000
            ),
        )

    async def generate_response(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Preserves the existing full-response inference method.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("LLM model is not loaded")

        if self.tokenizer is None:
            raise RuntimeError("LLM tokenizer is not loaded")

        def _generate() -> str:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(CUDA_DEVICE)

            prompt_token_count = (
                inputs["input_ids"].shape[-1]
            )

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=LLM_MAX_NEW_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    do_sample=LLM_TEMPERATURE > 0,
                    pad_token_id=(
                        self.tokenizer.eos_token_id
                    ),
                )

            response_tokens = generated[0][
                prompt_token_count:
            ]

            return self.tokenizer.decode(
                response_tokens,
                skip_special_tokens=True,
            ).strip()

        return await asyncio.to_thread(_generate)

    async def stream_response_chunks(
        self,
        messages: list[dict[str, str]],
    ):
        """
        Streams short sentence-sized chunks suitable for TTS.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError("LLM model is not loaded")

        if self.tokenizer is None:
            raise RuntimeError("LLM tokenizer is not loaded")

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(CUDA_DEVICE)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "do_sample": LLM_TEMPERATURE > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        generation_started_at = time.perf_counter()

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )

        thread.start()

        buffer = ""
        first_text_received = False
        chunk_index = 0

        logger.info(
            "LLM_STREAM started messages=%s",
            len(messages),
        )

        for text_piece in streamer:
            if (
                text_piece
                and not first_text_received
            ):
                first_text_received = True

                logger.info(
                    "LLM_STREAM first_text elapsed_ms=%s",
                    round(
                        (
                            time.perf_counter()
                            - generation_started_at
                        )
                        * 1000
                    ),
                )

            buffer += text_piece

            while True:
                chunk, buffer = self._extract_chunk(
                    buffer
                )

                if not chunk:
                    break

                chunk_index += 1

                logger.info(
                    "LLM_STREAM chunk_ready index=%s "
                    "elapsed_ms=%s words=%s text=%r",
                    chunk_index,
                    round(
                        (
                            time.perf_counter()
                            - generation_started_at
                        )
                        * 1000
                    ),
                    len(chunk.split()),
                    chunk,
                )

                yield chunk

            await asyncio.sleep(0)

        if buffer.strip():
            chunk_index += 1
            final_chunk = buffer.strip()

            logger.info(
                "LLM_STREAM chunk_ready index=%s "
                "elapsed_ms=%s words=%s text=%r",
                chunk_index,
                round(
                    (
                        time.perf_counter()
                        - generation_started_at
                    )
                    * 1000
                ),
                len(final_chunk.split()),
                final_chunk,
            )

            yield final_chunk

        logger.info(
            "LLM_STREAM complete chunks=%s elapsed_ms=%s",
            chunk_index,
            round(
                (
                    time.perf_counter()
                    - generation_started_at
                )
                * 1000
            ),
        )

    def _extract_chunk(
        self,
        buffer: str,
    ) -> tuple[str | None, str]:
        """
        Extracts a natural sentence boundary when available.

        If a sentence becomes too long, it is split near the configured
        word limit so TTS can begin sooner.
        """

        working = buffer.lstrip()

        if not working:
            return None, ""

        sentence_match = re.search(
            r'[.!?](?:["\']?)(?:\s|$)',
            working,
        )

        if sentence_match:
            end_index = sentence_match.end()

            candidate = working[:end_index].strip()

            if (
                len(candidate.split())
                <= LLM_CHUNK_MAX_WORDS
            ):
                remainder = working[
                    end_index:
                ].lstrip()

                return candidate, remainder

        word_matches = list(
            re.finditer(r"\S+", working)
        )

        if (
            len(word_matches)
            >= LLM_CHUNK_MAX_WORDS
        ):
            split_match = word_matches[
                LLM_CHUNK_MAX_WORDS - 1
            ]

            split_index = split_match.end()

            candidate = working[
                :split_index
            ].strip()

            remainder = working[
                split_index:
            ].lstrip()

            return candidate, remainder

        return None, buffer

    async def stream_sentences(
        self,
        user_text: str,
    ):
        """
        Compatibility wrapper for the existing websocket path.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Bailey, a concise realtime "
                    "voice assistant."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        async for chunk in self.stream_response_chunks(
            messages
        ):
            yield chunk