# BaileyAI Production Foundation v1

## requirements.txt

```txt
fastapi
uvicorn[standard]
python-dotenv
pydantic-settings
boto3
numpy
torch
transformers
accelerate
autoawq
faster-whisper
soundfile
python-multipart
websockets
jinja2
```

---

## .gitignore

```gitignore
.env
.venv/
__pycache__/
*.pyc
.idea/
.DS_Store
```

---

## .env.example

```env
APP_ENV=local
APP_NAME=BaileyAI
LOG_LEVEL=INFO

HOST=0.0.0.0
PORT=8000

CUDA_DEVICE=cuda

STT_MODEL_NAME=distil-whisper/distil-large-v3
STT_COMPUTE_TYPE=float16
STT_LANGUAGE=en

LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_MAX_NEW_TOKENS=120
LLM_TEMPERATURE=0.7

TTS_MODEL_NAME=chatterbox-turbo

AWS_REGION=us-east-1
SSM_PARAMETER_PREFIX=
```

---

## app/config.py

```python
import os
from functools import lru_cache
from typing import Optional

import boto3
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = Field(default="local")
    APP_NAME: str = Field(default="BaileyAI")
    LOG_LEVEL: str = Field(default="INFO")

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    CUDA_DEVICE: str = Field(default="cuda")

    STT_MODEL_NAME: str = Field(default="distil-whisper/distil-large-v3")
    STT_COMPUTE_TYPE: str = Field(default="float16")
    STT_LANGUAGE: str = Field(default="en")

    LLM_MODEL_NAME: str = Field(default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    LLM_MAX_NEW_TOKENS: int = Field(default=120)
    LLM_TEMPERATURE: float = Field(default=0.7)

    TTS_MODEL_NAME: str = Field(default="chatterbox-turbo")

    AWS_REGION: str = Field(default="us-east-1")
    SSM_PARAMETER_PREFIX: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


def load_ssm_parameters(prefix: str, region: str) -> None:
    client = boto3.client("ssm", region_name=region)

    next_token = None

    while True:
        kwargs = {
            "Path": prefix,
            "Recursive": True,
            "WithDecryption": True,
        }

        if next_token:
            kwargs["NextToken"] = next_token

        response = client.get_parameters_by_path(**kwargs)

        for param in response.get("Parameters", []):
            key = param["Name"].split("/")[-1]
            os.environ[key] = param["Value"]

        next_token = response.get("NextToken")

        if not next_token:
            break


@lru_cache
def get_settings() -> Settings:
    bootstrap = Settings()

    if bootstrap.APP_ENV != "local" and bootstrap.SSM_PARAMETER_PREFIX:
        load_ssm_parameters(
            prefix=bootstrap.SSM_PARAMETER_PREFIX,
            region=bootstrap.AWS_REGION,
        )

    return Settings()


settings = get_settings()
```

---

## app/logging_config.py

```python
import logging
import sys

from app.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
```

---

## app/model_manager.py

```python
from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService


class ModelManager:
    """
    Centralized GPU runtime manager.

    Responsible for:
    - loading models into VRAM
    - maintaining warm inference state
    - exposing singleton model services
    """

    def __init__(self):
        # Streaming speech-to-text runtime.
        self.stt = STTService()

        # Conversational language model runtime.
        self.llm = LLMService()

        # Text-to-speech runtime.
        self.tts = TTSService()

        # Indicates whether all models are loaded.
        self.loaded = False

    async def load_all(self):
        """
        Loads all inference runtimes into GPU VRAM.
        """

        # Load STT model.
        await self.stt.load()

        # Load conversational LLM.
        await self.llm.load()

        # Load TTS runtime.
        await self.tts.load()

        # Mark system ready.
        self.loaded = True

    def status(self):
        """
        Returns current model residency state.
        """

        return {
            "loaded": self.loaded,
            "stt": self.stt.loaded,
            "llm": self.llm.loaded,
            "tts": self.tts.loaded,
        }


# Global singleton instance.
model_manager = ModelManager()
```

---

## app/main.py

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.health import router as health_router
from app.api.startup import router as startup_router
from app.api.websocket import router as websocket_router
from app.config import settings
from app.logging_config import configure_logging

configure_logging()

app = FastAPI(title=settings.APP_NAME)

app.include_router(health_router)
app.include_router(startup_router)
app.include_router(websocket_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")
```

---

## app/api/health.py

```python
from fastapi import APIRouter

from app.model_manager import model_manager

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "models": model_manager.status(),
    }
```

---

## app/api/startup.py

```python
from fastapi import APIRouter

from app.model_manager import model_manager

router = APIRouter()


@router.post("/startup/load-models")
async def load_models():
    await model_manager.load_all()

    return {
        "status": "loaded",
        "models": model_manager.status(),
    }
```

---

## app/api/websocket.py

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.model_manager import model_manager

router = APIRouter()


@router.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    """
    Main realtime websocket handling browser audio streaming.
    """

    # Accept incoming websocket connection.
    await websocket.accept()

    # Prevent inference before GPU models are resident.
    if not model_manager.loaded:
        await websocket.send_json({
            "type": "error",
            "message": "Models not loaded"
        })

        await websocket.close()
        return

    # Stores incoming browser microphone chunks.
    audio_chunks = []

    try:
        while True:
            # Receive websocket payload.
            message = await websocket.receive()

            # Browser audio stream.
            if "bytes" in message and message["bytes"]:
                audio_chunks.append(message["bytes"])

            # Browser signals utterance completion.
            elif "text" in message and message["text"] == "end_utterance":

                # Run speech-to-text inference.
                transcript = await model_manager.stt.transcribe_webm(audio_chunks)

                # Clear audio buffer after transcription.
                audio_chunks.clear()

                # Return transcript to frontend.
                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                })

                # Stream sentence-by-sentence LLM output.
                async for sentence in model_manager.llm.stream_sentences(transcript):

                    # Send text immediately for responsive UI.
                    await websocket.send_json({
                        "type": "assistant_text",
                        "text": sentence,
                    })

                    # Generate TTS audio.
                    audio_bytes = await model_manager.tts.synthesize(sentence)

                    # Stream synthesized audio back to browser.
                    await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        # Browser disconnected.
        pass
```

---

## app/services/stt.py

```python
import asyncio
import tempfile

from faster_whisper import WhisperModel

from app.config import settings


class STTService:
    """
    Handles realtime speech-to-text inference using Faster-Whisper.
    """

    def __init__(self):
        self.model = None
        self.loaded = False

    async def load(self):
        """
        Loads Whisper weights into GPU VRAM.
        """

        def _load():
            return WhisperModel(
                settings.STT_MODEL_NAME,
                device=settings.CUDA_DEVICE,
                compute_type=settings.STT_COMPUTE_TYPE,
            )

        self.model = await asyncio.to_thread(_load)
        self.loaded = True

    async def transcribe_webm(self, chunks: list[bytes]) -> str:
        """
        Converts streamed browser audio into text.
        """

        audio_bytes = b"".join(chunks)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as f:
            f.write(audio_bytes)
            f.flush()

            def _transcribe():
                segments, _ = self.model.transcribe(
                    f.name,
                    beam_size=1,
                    vad_filter=True,
                    language=settings.STT_LANGUAGE,
                )

                return " ".join(seg.text.strip() for seg in segments)

            return await asyncio.to_thread(_transcribe)
```

---

## app/services/llm.py

```python
import asyncio
import threading

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from app.config import settings


class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    async def load(self):
        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL_NAME,
                trust_remote_code=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_NAME,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )

            return tokenizer, model

        self.tokenizer, self.model = await asyncio.to_thread(_load)
        self.loaded = True

    async def stream_sentences(self, user_text: str):
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
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
```

---

## app/services/tts.py

```python
import asyncio


class TTSService:
    def __init__(self):
        self.loaded = False

    async def load(self):
        self.loaded = True

    async def synthesize(self, text: str) -> bytes:
        await asyncio.sleep(0)
        return b""
```

---

## app/static/index.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>BaileyAI</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>

    <!-- Terms modal shown before application access -->
    <div id="termsModal" class="modal">
        <div class="modal-content">
            <h2>Terms & Agreement</h2>

            <p>
                By continuing, you consent to microphone access and realtime
                AI-generated voice interaction for testing purposes.
            </p>

            <button id="acceptTerms">Accept & Continue</button>
        </div>
    </div>

    <div class="container hidden" id="appContainer">
        <h1>BaileyAI Voice Demo</h1>

        <div class="controls">
            <button id="loadModels">Load Models</button>
            <button id="connect">Connect</button>
            <button id="start">🎤 Start Conversation</button>
            <button id="stop">Stop</button>
        </div>

        <p id="status">Idle</p>

        <div class="panel">
            <h3>You</h3>
            <p id="transcript"></p>
        </div>

        <div class="panel">
            <h3>Bailey</h3>
            <p id="assistant"></p>
        </div>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
```

---

## app/static/styles.css

```css
body {
    margin: 0;
    background: #0f1115;
    color: white;
    font-family: Arial, sans-serif;
}

.hidden {
    display: none;
}

.container {
    max-width: 900px;
    margin: auto;
    padding: 40px;
}

.controls {
    margin-bottom: 20px;
}

button {
    margin-right: 10px;
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}

.panel {
    background: #1a1d24;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

.modal {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.8);
    display: flex;
    justify-content: center;
    align-items: center;
}

.modal-content {
    background: #1a1d24;
    padding: 30px;
    border-radius: 12px;
    width: 500px;
}
```

---

## app/static/app.js

```javascript
// Primary websocket connection.
let ws;

// Browser media recorder for microphone streaming.
let mediaRecorder;

// Active microphone stream.
let stream;

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const assistantEl = document.getElementById("assistant");

const modalEl = document.getElementById("termsModal");
const appContainerEl = document.getElementById("appContainer");


// Enables application after terms acceptance.
document.getElementById("acceptTerms").onclick = () => {
    modalEl.style.display = "none";
    appContainerEl.classList.remove("hidden");
};


// Loads all inference runtimes into GPU VRAM.
document.getElementById("loadModels").onclick = async () => {
    const response = await fetch("/startup/load-models", {
        method: "POST",
    });

    const data = await response.json();

    statusEl.textContent = JSON.stringify(data);
};


// Opens websocket connection to realtime backend.
document.getElementById("connect").onclick = async () => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";

    ws = new WebSocket(`${protocol}://${window.location.host}/ws/voice`);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        statusEl.textContent = "Connected";
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === "string") {
            const msg = JSON.parse(event.data);

            if (msg.type === "transcript") {
                transcriptEl.textContent = msg.text;
            }

            if (msg.type === "assistant_text") {
                assistantEl.textContent += " " + msg.text;
            }
        } else {
            const blob = new Blob([event.data], { type: "audio/wav" });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.play();
        }
    };
};


// Starts microphone capture and streams chunks to backend.
document.getElementById("start").onclick = async () => {
    transcriptEl.textContent = "";
    assistantEl.textContent = "";

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm"
    });

    mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            const buffer = await event.data.arrayBuffer();
            ws.send(buffer);
        }
    };

    mediaRecorder.start(250);

    statusEl.textContent = "Recording";
};


// Stops recording and signals utterance completion.
document.getElementById("stop").onclick = async () => {
    mediaRecorder.stop();

    stream.getTracks().forEach(track => track.stop());

    ws.send("end_utterance");

    statusEl.textContent = "Processing";
};
```

---

## scripts/dev.sh

```bash
#!/bin/bash

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## scripts/startup.sh

```bash
#!/bin/bash

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## docker/Dockerfile

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## docker/docker-compose.yml

```yaml
version: '3.9'

services:
  baileyai:
    build:
      context: ..
      dockerfile: docker/Dockerfile

    ports:
      - "8000:8000"

    env_file:
      - ../.env

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## test_main.http

```http
### Health
GET http://127.0.0.1:8000/health

### Load Models
POST http://127.0.0.1:8000/startup/load-models
```

---

# What I Need From You To Deploy/Test

## Required immediately

```text
1. Create .env from .env.example
2. pip install -r requirements.txt
3. chmod +x scripts/dev.sh
4. Run: ./scripts/dev.sh
5. Open: http://127.0.0.1:8000
```

## Needed before live GPU deployment

```text
1. AWS EC2 g6e.2xlarge provisioned
2. NVIDIA drivers installed
3. Docker + NVIDIA Container Toolkit installed
4. Exact Chatterbox-Turbo runtime package/API finalized
5. HuggingFace access token if gated models are used
6. SSL domain/cert once HTTPS/WSS is enabled
7. SSM Parameter Store prefix path
```

# Current Known Placeholder

```text
app/services/tts.py
```

That file is intentionally isolated because we still need the exact Chatterbox-Turbo inference runtime/package you want standardized.

