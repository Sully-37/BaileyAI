from fastapi import APIRouter, HTTPException

from app.model_manager import model_manager

router = APIRouter()


@router.get("/health")
async def health():
    """
    Confirms the API process is alive.
    """
    return {
        "status": "ok",
        "models": model_manager.status(),
    }


@router.get("/ready")
async def ready():
    """
    Confirms inference models are loaded and ready.
    """
    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": "Models are not loaded",
                "models": model_manager.status(),
            },
        )

    return {
        "status": "ready",
        "models": model_manager.status(),
    }