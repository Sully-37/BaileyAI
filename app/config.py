"""
Application configuration.
"""

APP_NAME = "BaileyAI"

HOST = "0.0.0.0"
PORT = 8000

LOG_LEVEL = "INFO"

CUDA_DEVICE = "cuda"

# Speech-to-Text

STT_MODEL_NAME = "Systran/faster-distil-whisper-large-v3"
STT_COMPUTE_TYPE = "float16"
STT_LANGUAGE = "en"
STT_VAD_MIN_SILENCE_MS = 300

# Language Model

LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_NEW_TOKENS = 120
LLM_TEMPERATURE = 0.7

# Streaming response tuning.
LLM_INITIAL_BUFFER_CHUNKS = 1
LLM_CHUNK_MAX_WORDS = 18

# Text-to-Speech

TTS_MODEL_NAME = "chatterbox-turbo"
VOICE_REFERENCE_PATH = "app/voices/bailey_reference.wav"

# Runs once during model loading so the user's first real
# TTS request does not pay the cold-start inference penalty.
TTS_WARMUP_TEXT = "Ready."
