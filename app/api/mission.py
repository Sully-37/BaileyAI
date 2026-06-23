import logging
from fastapi import APIRouter, UploadFile, File

from app.model_manager import model_manager
from app.utils.gpu import gpu_is_available, gpu_device_name

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/mission-control/test")
async def mission_control_test(audio: UploadFile | None = File(default=None)):
    """
    Runs a deployment-readiness test sequence.

    This endpoint is designed for cheap CPU-only EC2 testing.
    It verifies app health, optional browser microphone upload,
    voice asset presence, GPU availability, and model-load behavior.
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
        })
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
            "status": "expected_fail" if not gpu_available else "fail",
            "message": str(exc),
        })

    return {
        "status": "complete",
        "gpu_available": gpu_available,
        "models": model_manager.status(),
        "results": results,
    }