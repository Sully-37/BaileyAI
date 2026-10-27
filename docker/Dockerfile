FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        sox \
        libsox-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install BaileyAI dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Validate installed dependencies.
RUN pip check

# Verify the core PyTorch/CUDA runtime.
RUN python -c "\
import os, torch, torchaudio, transformers; \
print('torch:', torch.__version__); \
print('torchaudio:', torchaudio.__version__); \
print('transformers:', transformers.__version__); \
print('torch cuda:', torch.version.cuda); \
print('supported architectures:', torch.cuda.get_arch_list()); \
print('CUDA_HOME:', os.environ.get('CUDA_HOME')); \
print('cuda available during build:', torch.cuda.is_available())"

# Verify CUDA compiler.
RUN nvcc --version

# Copy BaileyAI into the image.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]