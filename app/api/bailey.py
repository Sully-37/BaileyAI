import base64
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.model_manager import model_manager
from app.session_store import session_store

router = APIRouter(prefix="/bailey", tags=["bailey"])
logger = logging.getLogger(__name__)


def elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


@router.post("/session")
async def create_session():
    """
    Creates an in-memory Bailey conversation.
    """

    session = await session_store.create()

    return {
        "status": "created",
        "session_id": session.session_id,
        "expires_in_seconds": session_store.ttl_seconds,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Deletes all in-memory context associated with the session.
    """

    deleted = await session_store.delete(session_id)

    return {
        "status": "deleted" if deleted else "not_found",
        "session_id": session_id,
    }


@router.get("/status")
async def bailey_status():
    """
    Returns whether Bailey can accept a conversation turn.
    """

    return {
        "status": "ready" if model_manager.loaded else "not_ready",
        "ready": model_manager.loaded,
        "models": model_manager.status(),
    }


@router.post("/turn")
async def bailey_turn(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Runs one complete Bailey conversation turn:

    audio -> STT -> session context -> LLM -> TTS -> browser audio
    """

    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": "Bailey's models are not loaded.",
                "models": model_manager.status(),
            },
        )

    session = await session_store.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "session_not_found",
                "message": "The conversation session expired or does not exist.",
            },
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "invalid_audio",
                "message": "No microphone audio was received.",
            },
        )

    timings: dict[str, int] = {}
    total_started_at = time.perf_counter()

    async with session.lock:
        try:
            stt_started_at = time.perf_counter()

            transcript = await model_manager.stt.transcribe_bytes(
                audio_bytes=audio_bytes,
                suffix=".webm",
            )

            timings["stt_ms"] = elapsed_ms(stt_started_at)

            llm_messages = [
                *session.messages,
                {
                    "role": "user",
                    "content": transcript,
                },
            ]

            llm_started_at = time.perf_counter()

            response_text = await model_manager.llm.generate_response(
                llm_messages
            )

            timings["llm_ms"] = elapsed_ms(llm_started_at)

            tts_started_at = time.perf_counter()

            audio_response = await model_manager.tts.synthesize(
                response_text
            )

            timings["tts_ms"] = elapsed_ms(tts_started_at)

            session_store.append_turn(
                session=session,
                user_text=transcript,
                assistant_text=response_text,
            )

            timings["total_ms"] = elapsed_ms(total_started_at)

            logger.info(
                "Bailey turn complete session=%s turn=%s "
                "stt_ms=%s llm_ms=%s tts_ms=%s total_ms=%s",
                session.session_id,
                session.turn_count,
                timings["stt_ms"],
                timings["llm_ms"],
                timings["tts_ms"],
                timings["total_ms"],
            )

            return {
                "status": "complete",
                "session_id": session.session_id,
                "turn": session.turn_count,
                "transcript": transcript,
                "response_text": response_text,
                "audio_mime_type": "audio/wav",
                "audio_base64": base64.b64encode(
                    audio_response
                ).decode("ascii"),
                "timings": timings,
            }

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Bailey conversation turn failed for session %s",
                session_id,
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "message": str(exc),
                    "timings": timings,
                },
            ) from exc