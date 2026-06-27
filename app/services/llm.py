import asyncio
import threading

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from app.config import (
    CUDA_DEVICE,
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
)


class LLMService:
    """
    GPU resident conversational language model runtime.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    async def load(self):
        """
        Loads the quantized Qwen model into GPU memory.
        """

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                device_map=CUDA_DEVICE,
                torch_dtype="auto",
                trust_remote_code=True,
            )

            return tokenizer, model

        self.tokenizer, self.model = await asyncio.to_thread(_load)
        self.loaded = True

    async def stream_sentences(self, user_text: str):
        """
        Streams model output as sentence-sized chunks for TTS.
        """

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

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True,
        )

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )

        thread.start()

        buffer = ""

        for token in streamer:
            buffer += token

            if any(buffer.endswith(p) for p in [".", "!", "?", "\n"]):
                sentence = buffer.strip()
                buffer = ""

                if sentence:
                    yield sentence

            await asyncio.sleep(0)

        if buffer.strip():
            yield buffer.strip()