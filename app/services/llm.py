import asyncio
import threading
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from app.config import (
    CUDA_DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)
from app.utils.gpu import gpu_is_available


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
        Loads the quantized Qwen model into GPU memory.
        """

        if self.loaded:
            return

        if not gpu_is_available():
            raise RuntimeError(
                "GPU unavailable. Qwen inference requires CUDA before model loading."
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

        self.tokenizer, self.model = await asyncio.to_thread(_load)
        self.loaded = True

    async def generate_response(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generates one complete response using the provided conversation history.
        """

        if not self.loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model is not loaded")

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

            generated = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=LLM_TEMPERATURE > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            prompt_token_count = inputs["input_ids"].shape[-1]
            response_tokens = generated[0][prompt_token_count:]

            return self.tokenizer.decode(
                response_tokens,
                skip_special_tokens=True,
            ).strip()

        response = await asyncio.to_thread(_generate)

        if not response:
            raise RuntimeError("LLM returned an empty response")

        return response

    async def stream_sentences(self, user_text: str):
        """
        Preserves the existing sentence-streaming interface used by the
        websocket implementation.
        """

        if not self.loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model is not loaded")

        messages = [
            {
                "role": "system",
                "content": "You are Bailey, a concise realtime voice assistant.",
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

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

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )

        thread.start()

        buffer = ""

        for token in streamer:
            buffer += token

            if any(buffer.endswith(mark) for mark in [".", "!", "?", "\n"]):
                sentence = buffer.strip()
                buffer = ""

                if sentence:
                    yield sentence

            await asyncio.sleep(0)

        if buffer.strip():
            yield buffer.strip()