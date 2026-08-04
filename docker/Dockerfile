FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

# Fail the image build immediately if the PyTorch binary set is broken.
RUN python -c "\
import torch, torchvision, torchaudio; \
print('torch:', torch.__version__); \
print('torchvision:', torchvision.__version__); \
print('torchaudio:', torchaudio.__version__)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]