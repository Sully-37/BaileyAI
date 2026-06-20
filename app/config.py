"""
Application configuration.

These values define the architecture and are intentionally
stored in source control.

No secrets are currently required.
"""

APP_NAME = "BaileyAI"

HOST = "0.0.0.0"
PORT = 8000

LOG_LEVEL = "INFO"

CUDA_DEVICE = "cuda"

# Speech-to-Text

STT_MODEL_NAME = "distil-whisper/distil-large-v3"
STT_COMPUTE_TYPE = "float16"
STT_LANGUAGE = "en"

# Language Model

LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
LLM_MAX_NEW_TOKENS = 120
LLM_TEMPERATURE = 0.7

# Text-to-Speech

TTS_MODEL_NAME = "chatterbox-turbo"

VOICE_REFERENCE_PATH = "app/voices/bailey_reference.wav"