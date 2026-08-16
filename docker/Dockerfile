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

RUN pip install --no-cache-dir -r requirements.txt

# Make CosyVoice and Matcha-TTS importable from Bailey.
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

WORKDIR /app

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