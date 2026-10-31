import asyncio
import base64
import json
import logging
import queue
import time
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from app.model_manager import model_manager
from app.session_store import session_store

router = APIRouter(
    prefix="/bailey",
    tags=["bailey"],
)

logger = logging.getLogger(__name__)


def elapsed_ms(started_at: float) -> int:
    """
    Returns elapsed milliseconds from a perf_counter start.
    """

    return round(
        (time.perf_counter() - started_at)
        * 1000
    )


def utc_timestamp() -> str:
    """UTC wall-clock timestamp for correlating browser and server events."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def stream_event(payload: dict) -> str:
    """
    Encodes one newline-delimited JSON event.
    """

    return (
        json.dumps(
            payload,
            separators=(",", ":"),
        )
        + "\n"
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
        "expires_in_seconds": (
            session_store.ttl_seconds
        ),
    }


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
):
    """
    Deletes in-memory session context.
    """

    deleted = await session_store.delete(
        session_id
    )

    logger.info(
        "SESSION deleted session=%s existed=%s",
        session_id,
        deleted,
    )

    return {
        "status": (
            "deleted"
            if deleted
            else "not_found"
        ),
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
    Runs one Bailey conversation turn.

    Browser audio
    -> STT
    -> streaming Qwen
    -> continuous CosyVoice bi-stream
    -> streamed browser audio
    """

    request_id = uuid.uuid4().hex[:8]
    request_received_at = time.perf_counter()
    request_received_utc = utc_timestamp()

    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": (
                    "Bailey's models are not loaded."
                ),
                "models": model_manager.status(),
            },
        )

    session = await session_store.get(
        session_id
    )

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

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "invalid_audio",
                "message": (
                    "No microphone audio was received."
                ),
            },
        )

    logger.info(
        "TURN RECEIVED request=%s at_utc=%s audio_bytes=%s upload_read_ms=%s",
        request_id,
        request_received_utc,
        len(audio_bytes),
        elapsed_ms(request_received_at),
    )

    async def generate_stream():
        total_started_at = (
            time.perf_counter()
        )

        timings = {
            "request_received_utc": request_received_utc,
            "upload_read_ms": elapsed_ms(request_received_at),
        }

        async with session.lock:
            timings["session_lock_wait_ms"] = elapsed_ms(total_started_at)
            llm_task = None

            try:
                # ----------------------------------
                # STT
                # ----------------------------------

                stt_started_at = time.perf_counter()
                timings["stt_started_utc"] = utc_timestamp()
                logger.info("STT START request=%s at_utc=%s", request_id, timings["stt_started_utc"])

                transcript = (
                    await model_manager.stt
                    .transcribe_bytes(
                        audio_bytes=audio_bytes,
                        suffix=".webm",
                    )
                )

                timings["stt_ms"] = (
                    elapsed_ms(
                        stt_started_at
                    )
                )

                timings["stt_completed_utc"] = utc_timestamp()
                logger.info(
                    "STT COMPLETE request=%s at_utc=%s duration_ms=%s",
                    request_id, timings["stt_completed_utc"], timings["stt_ms"],
                )

                yield stream_event({
                    "type": "transcript",
                    "request_id": request_id,
                    "text": transcript,
                    "timings": {
                        "stt_ms": (
                            timings["stt_ms"]
                        ),
                    },
                })

                # ----------------------------------
                # Conversation context
                # ----------------------------------

                llm_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Bailey, a concise conversational voice assistant. "
                            "Never use exclamation marks (!) in any response. "
                            "Use calm, natural language suitable for text-to-speech. "
                            "Avoid exaggerated emphasis."
                        ),
                    },
                    *session.messages,
                    {
                        "role": "user",
                        "content": transcript,
                    },
                ]

                # Thread-safe bridge between Qwen's
                # async producer and CosyVoice's
                # synchronous text generator.
                text_queue = queue.Queue()

                llm_chunks: list[str] = []

                llm_error = None
                first_chunk_seen = False

                llm_started_at = time.perf_counter()
                timings["llm_started_utc"] = utc_timestamp()
                logger.info("LLM START request=%s at_utc=%s", request_id, timings["llm_started_utc"])

                # ----------------------------------
                # Qwen producer
                # ----------------------------------

                async def produce_llm():
                    nonlocal llm_error
                    nonlocal first_chunk_seen

                    try:
                        async for chunk in (
                            model_manager.llm
                            .stream_response_chunks(
                                llm_messages
                            )
                        ):
                            if (
                                not first_chunk_seen
                            ):
                                first_chunk_seen = (
                                    True
                                )

                                timings[
                                    "llm_first_chunk_ms"
                                ] = elapsed_ms(
                                    llm_started_at
                                )

                                timings["llm_first_chunk_utc"] = utc_timestamp()
                                logger.info(
                                    "LLM FIRST CHUNK request=%s at_utc=%s duration_ms=%s",
                                    request_id,
                                    timings["llm_first_chunk_utc"],
                                    timings["llm_first_chunk_ms"],
                                )

                            llm_chunks.append(
                                chunk
                            )

                            # Feed this chunk directly
                            # into the live CosyVoice
                            # inference.
                            text_queue.put(
                                chunk
                            )

                    except Exception as exc:
                        llm_error = exc

                        text_queue.put(
                            exc
                        )

                    finally:
                        timings[
                            "llm_total_ms"
                        ] = elapsed_ms(
                            llm_started_at
                        )

                        # Signals CosyVoice that Qwen
                        # has finished producing text.
                        text_queue.put(None)

                llm_task = asyncio.create_task(
                    produce_llm()
                )

                # ----------------------------------
                # ONE continuous CosyVoice inference
                # ----------------------------------

                tts_started_at = time.perf_counter()
                timings["tts_started_utc"] = utc_timestamp()
                logger.info("TTS STREAM START request=%s at_utc=%s", request_id, timings["tts_started_utc"])

                first_audio_ready = False
                audio_chunk_index = 0

                async for audio_response in (
                    model_manager.tts
                    .stream_audio_from_text_queue(
                        text_queue
                    )
                ):
                    audio_chunk_index += 1

                    if not first_audio_ready:
                        first_audio_ready = True

                        timings[
                            "first_audio_ready_ms"
                        ] = elapsed_ms(
                            total_started_at
                        )

                        timings[
                            "tts_first_audio_ms"
                        ] = elapsed_ms(
                            tts_started_at
                        )

                        timings["first_audio_ready_utc"] = utc_timestamp()
                        logger.info(
                            "FIRST AUDIO READY request=%s at_utc=%s "
                            "tts_stream_ms=%s backend_ms=%s",
                            request_id,
                            timings["first_audio_ready_utc"],
                            timings["tts_first_audio_ms"],
                            timings["first_audio_ready_ms"],
                        )

                    yield stream_event({
                        "type": "audio",
                        "request_id": request_id,
                        "index": (
                            audio_chunk_index
                        ),
                        "audio_mime_type": (
                            "audio/pcm"
                        ),
                        "sample_rate": (
                            model_manager.tts.sample_rate
                        ),
                        "audio_base64": (
                            base64.b64encode(
                                audio_response
                            ).decode("ascii")
                        ),
                        "timings": {
                            "tts_elapsed_ms": (
                                elapsed_ms(
                                    tts_started_at
                                )
                            ),
                            "backend_elapsed_ms": (
                                elapsed_ms(
                                    total_started_at
                                )
                            ),
                        },
                    })

                timings["tts_total_ms"] = (
                    elapsed_ms(
                        tts_started_at
                    )
                )

                # ----------------------------------
                # Finish Qwen
                # ----------------------------------

                await llm_task

                if llm_error:
                    raise llm_error

                response_text = " ".join(
                    llm_chunks
                ).strip()

                if not response_text:
                    raise RuntimeError(
                        "LLM returned an empty response"
                    )

                # ----------------------------------
                # Save in-memory context
                # ----------------------------------

                session_store.append_turn(
                    session=session,
                    user_text=transcript,
                    assistant_text=response_text,
                )

                timings[
                    "backend_stream_complete_ms"
                ] = elapsed_ms(
                    total_started_at
                )

                timings["backend_completed_utc"] = utc_timestamp()
                logger.info(
                    "TURN COMPLETE request=%s at_utc=%s turn=%s "
                    "stt_ms=%s llm_first_chunk_ms=%s llm_total_ms=%s "
                    "tts_stream_first_audio_ms=%s tts_stream_total_ms=%s "
                    "first_audio_ready_ms=%s backend_total_ms=%s "
                    "upload_read_ms=%s session_lock_wait_ms=%s",
                    request_id,
                    timings["backend_completed_utc"],
                    session.turn_count,
                    timings.get("stt_ms"),
                    timings.get("llm_first_chunk_ms"),
                    timings.get("llm_total_ms"),
                    timings.get("tts_first_audio_ms"),
                    timings.get("tts_total_ms"),
                    timings.get("first_audio_ready_ms"),
                    timings["backend_stream_complete_ms"],
                    timings["upload_read_ms"],
                    timings["session_lock_wait_ms"],
                )

                yield stream_event({
                    "type": "complete",
                    "request_id": request_id,
                    "session_id": (
                        session.session_id
                    ),
                    "turn": session.turn_count,
                    "response_text": (
                        response_text
                    ),
                    "timings": timings,
                })

            except Exception as exc:
                logger.exception(
                    "TURN failed request=%s "
                    "session=%s error=%s",
                    request_id,
                    session_id,
                    exc,
                )

                if (
                    llm_task
                    and not llm_task.done()
                ):
                    llm_task.cancel()

                yield stream_event({
                    "type": "error",
                    "request_id": request_id,
                    "message": str(exc),
                    "timings": timings,
                })

    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
        },
    )
