import base64
import logging
import time
import uuid

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.model_manager import model_manager
from app.session_store import session_store

router = APIRouter(
    prefix="/bailey",
    tags=["bailey"],
)

logger = logging.getLogger(__name__)


def elapsed_ms(started_at: float) -> int:
    return round(
        (time.perf_counter() - started_at) * 1000
    )


@router.post("/session")
async def create_session():
    """
    Creates an in-memory Bailey conversation.
    """

    session = await session_store.create()

    logger.info(
        "SESSION created session=%s",
        session.session_id,
    )

    return {
        "status": "created",
        "session_id": session.session_id,
        "expires_in_seconds": session_store.ttl_seconds,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Deletes in-memory session context.
    """

    deleted = await session_store.delete(session_id)

    logger.info(
        "SESSION deleted session=%s existed=%s",
        session_id,
        deleted,
    )

    return {
        "status": "deleted" if deleted else "not_found",
        "session_id": session_id,
    }


@router.get("/status")
async def bailey_status():
    """
    Returns whether Bailey can accept a turn.
    """

    return {
        "status": (
            "ready"
            if model_manager.loaded
            else "not_ready"
        ),
        "ready": model_manager.loaded,
        "models": model_manager.status(),
    }


@router.post("/turn")
async def bailey_turn(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Executes one complete voice turn.
    """

    request_id = uuid.uuid4().hex[:8]
    total_started_at = time.perf_counter()

    logger.info(
        "TURN started request=%s session=%s "
        "filename=%s content_type=%s",
        request_id,
        session_id,
        audio.filename,
        audio.content_type,
    )

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
                "message": (
                    "The conversation session expired "
                    "or does not exist."
                ),
            },
        )

    timings: dict[str, int] = {}

    audio_read_started_at = time.perf_counter()
    audio_bytes = await audio.read()
    timings["audio_read_ms"] = elapsed_ms(
        audio_read_started_at
    )

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "invalid_audio",
                "message": "No microphone audio was received.",
            },
        )

    logger.info(
        "TURN audio_received request=%s bytes=%s "
        "audio_read_ms=%s",
        request_id,
        len(audio_bytes),
        timings["audio_read_ms"],
    )

    async with session.lock:
        try:
            stt_started_at = time.perf_counter()

            transcript = await model_manager.stt.transcribe_bytes(
                audio_bytes=audio_bytes,
                suffix=".webm",
            )

            timings["stt_ms"] = elapsed_ms(stt_started_at)

            logger.info(
                "TURN stt_complete request=%s elapsed_ms=%s "
                "transcript=%r",
                request_id,
                timings["stt_ms"],
                transcript,
            )

            llm_messages = [
                *session.messages,
                {
                    "role": "user",
                    "content": transcript,
                },
            ]

            llm_started_at = time.perf_counter()

            response_text = (
                await model_manager.llm.generate_response(
                    llm_messages
                )
            )

            timings["llm_ms"] = elapsed_ms(llm_started_at)

            logger.info(
                "TURN llm_complete request=%s elapsed_ms=%s "
                "response_chars=%s",
                request_id,
                timings["llm_ms"],
                len(response_text),
            )

            tts_started_at = time.perf_counter()

            audio_response = await model_manager.tts.synthesize(
                response_text
            )

            timings["tts_ms"] = elapsed_ms(tts_started_at)

            logger.info(
                "TURN tts_complete request=%s elapsed_ms=%s "
                "audio_bytes=%s",
                request_id,
                timings["tts_ms"],
                len(audio_response),
            )

            session_store.append_turn(
                session=session,
                user_text=transcript,
                assistant_text=response_text,
            )

            encode_started_at = time.perf_counter()

            encoded_audio = base64.b64encode(
                audio_response
            ).decode("ascii")

            timings["audio_encode_ms"] = elapsed_ms(
                encode_started_at
            )

            timings["backend_total_ms"] = elapsed_ms(
                total_started_at
            )

            logger.info(
                "TURN complete request=%s session=%s turn=%s "
                "audio_read_ms=%s stt_ms=%s llm_ms=%s "
                "tts_ms=%s encode_ms=%s backend_total_ms=%s",
                request_id,
                session.session_id,
                session.turn_count,
                timings["audio_read_ms"],
                timings["stt_ms"],
                timings["llm_ms"],
                timings["tts_ms"],
                timings["audio_encode_ms"],
                timings["backend_total_ms"],
            )

            return {
                "status": "complete",
                "request_id": request_id,
                "session_id": session.session_id,
                "turn": session.turn_count,
                "transcript": transcript,
                "response_text": response_text,
                "audio_mime_type": "audio/wav",
                "audio_base64": encoded_audio,
                "timings": timings,
            }

        except HTTPException:
            raise

        except Exception as exc:
            timings["backend_total_ms"] = elapsed_ms(
                total_started_at
            )

            logger.exception(
                "TURN failed request=%s session=%s "
                "elapsed_ms=%s error=%s",
                request_id,
                session_id,
                timings["backend_total_ms"],
                exc,
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "request_id": request_id,
                    "message": str(exc),
                    "timings": timings,
                },
            ) from exc