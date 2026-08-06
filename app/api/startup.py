from fastapi import APIRouter, HTTPException

from app.model_manager import model_manager
from app.utils.gpu import gpu_is_available

router = APIRouter()


@router.post("/startup/load-models")
async def load_models():
    """
    Starts model loading asynchronously.
    """

    if not gpu_is_available():
        raise HTTPException(
            status_code=503,
            detail={
                "status": "gpu_unavailable",
                "message": (
                    "GPU unavailable. This server can deploy the app "
                    "but cannot load inference models."
                ),
                "models": model_manager.status(),
            },
        )

    await model_manager.start_loading()

    return {
        "status": "loaded" if model_manager.loaded else "loading",
        "models": model_manager.status(),
    }
