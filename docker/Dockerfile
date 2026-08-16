FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        sox \
        libsox-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install CosyVoice and its required submodules.
RUN git clone --recursive \
    https://github.com/QwenAudio/CosyVoice.git \
    /opt/CosyVoice

WORKDIR /opt/CosyVoice

# Whisper requires pkg_resources during its build.
RUN pip install --no-cache-dir "setuptools<70" wheel

# Install Whisper separately without build isolation.
RUN pip install --no-cache-dir \
    --no-build-isolation \
    openai-whisper==20231117

# Install the rest of CosyVoice normally.
RUN grep -v "^openai-whisper==" requirements.txt \
    > /tmp/cosyvoice-requirements.txt && \
    pip install --no-cache-dir \
    -r /tmp/cosyvoice-requirements.txt

# Make CosyVoice and Matcha-TTS importable from Bailey.
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS"

WORKDIR /app

# Install Bailey's own dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Verify the core runtime.
RUN python -c "\
import torch, torchaudio, transformers; \
print('torch:', torch.__version__); \
print('torchaudio:', torchaudio.__version__); \
print('transformers:', transformers.__version__); \
print('cuda:', torch.cuda.is_available())"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]