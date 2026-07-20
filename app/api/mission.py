import json
import logging
import os
import subprocess
import tempfile

from fastapi import APIRouter, File, UploadFile

from app.model_manager import model_manager
from app.utils.gpu import gpu_device_name, gpu_is_available

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/mission-control/test")
async def mission_control_test(audio: UploadFile | None = File(default=None)):
    """
    Runs deployment-readiness checks without requiring a GPU.

    Validates:
    - API process is reachable
    - optional browser microphone audio reaches backend
    - audio payload can be inspected by ffprobe
    - GPU availability is correctly detected
    - model loading is skipped on CPU-only servers
    """

    results = []

    results.append({
        "step": "api_process",
        "status": "pass",
        "message": "FastAPI process is reachable.",
    })

    if audio:
        audio_bytes = await audio.read()

        results.append({
            "step": "browser_microphone_audio",
            "status": "pass" if len(audio_bytes) > 0 else "fail",
            "message": f"Received {len(audio_bytes)} bytes from browser microphone.",
            "content_type": audio.content_type,
            "filename": audio.filename,
        })

        audio_probe_result = _inspect_audio(audio_bytes)

        results.append(audio_probe_result)

    else:
        results.append({
            "step": "browser_microphone_audio",
            "status": "skipped",
            "message": "No microphone audio submitted.",
        })

    gpu_available = gpu_is_available()

    results.append({
        "step": "gpu_check",
        "status": "pass" if gpu_available else "expected_fail",
        "message": gpu_device_name() if gpu_available else "No CUDA GPU available on this server.",
    })

    if not gpu_available:
        results.append({
            "step": "model_load",
            "status": "expected_fail",
            "message": "Skipped model loading because no CUDA GPU is available.",
        })

        return {
            "status": "complete",
            "gpu_available": False,
            "models": model_manager.status(),
            "results": results,
        }

    try:
        await model_manager.load_all()

        results.append({
            "step": "model_load",
            "status": "pass",
            "message": "Models loaded into GPU memory.",
        })

    except Exception as exc:
        logger.exception("Mission control model-load test failed.")

        results.append({
            "step": "model_load",
            "status": "fail",
            "message": str(exc),
        })

    return {
        "status": "complete",
        "gpu_available": gpu_available,
        "models": model_manager.status(),
        "results": results,
    }


def _inspect_audio(audio_bytes: bytes) -> dict:
    """
    Writes uploaded browser audio to a temp file and validates that ffprobe can inspect it.
    """

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            temp_path,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            return {
                "step": "audio_decode_ready",
                "status": "fail",
                "message": result.stderr.strip() or "ffprobe could not inspect audio.",
            }

        probe = json.loads(result.stdout)

        return {
            "step": "audio_decode_ready",
            "status": "pass",
            "message": "Audio payload is readable and ready for STT handoff.",
            "format": probe.get("format", {}).get("format_name"),
            "duration": probe.get("format", {}).get("duration"),
            "streams": [
                {
                    "codec_type": stream.get("codec_type"),
                    "codec_name": stream.get("codec_name"),
                    "sample_rate": stream.get("sample_rate"),
                    "channels": stream.get("channels"),
                }
                for stream in probe.get("streams", [])
            ],
        }

    except Exception as exc:
        logger.exception("Audio inspection failed.")

        return {
            "step": "audio_decode_ready",
            "status": "fail",
            "message": str(exc),
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)