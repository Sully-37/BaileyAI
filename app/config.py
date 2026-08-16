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

LLM_INITIAL_BUFFER_CHUNKS = 1
LLM_CHUNK_MAX_WORDS = 18

# Text-to-Speech

TTS_MODEL_NAME = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

TTS_MODEL_PATH = (
    "/opt/bailey-models/Fun-CosyVoice3-0.5B"
)

VOICE_REFERENCE_PATH = (
    "app/voices/bailey_reference.wav"
)

# IMPORTANT:
# Replace this with the exact words spoken in
# bailey_reference.wav.
VOICE_REFERENCE_TEXT = (
    "REPLACE WITH EXACT BAILEY REFERENCE TRANSCRIPT"
)

TTS_ZERO_SHOT_SPEAKER_ID = "bailey"

TTS_WARMUP_TEXT = "Ready."