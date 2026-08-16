import asyncio
import base64
import json
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
from fastapi.responses import StreamingResponse

from app.config import LLM_INITIAL_BUFFER_CHUNKS
from app.model_manager import model_manager
from app.session_store import session_store

router = APIRouter(
    prefix="/bailey",
    tags=["bailey"],
)

logger = logging.getLogger(__name__)


def elapsed_ms(started_at: float) -> int:
    """
    Returns elapsed milliseconds from a perf_counter start time.
    """

    return round(
        (time.perf_counter() - started_at)
        * 1000
    )


def stream_event(payload: dict) -> str:
    """
    Encodes one newline-delimited JSON streaming event.
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
    Streams one complete Bailey voice turn.

    Flow:

    browser audio
    -> STT
    -> contextual LLM streaming
    -> short text chunks
    -> CosyVoice streaming TTS
    -> streamed browser audio
    """

    request_id = uuid.uuid4().hex[:8]

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

    audio_read_started_at = time.perf_counter()

    audio_bytes = await audio.read()

    audio_read_ms = elapsed_ms(
        audio_read_started_at
    )

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
        "TURN started request=%s session=%s "
        "audio_bytes=%s",
        request_id,
        session_id,
        len(audio_bytes),
    )

    async def generate_stream():
        """
        Produces newline-delimited JSON events for the browser.
        """

        total_started_at = time.perf_counter()

        timings = {
            "audio_read_ms": audio_read_ms,
        }

        async with session.lock:
            try:
                # -----------------------------------------
                # Speech-to-Text
                # -----------------------------------------

                stt_started_at = time.perf_counter()

                transcript = (
                    await model_manager.stt.transcribe_bytes(
                        audio_bytes=audio_bytes,
                        suffix=".webm",
                    )
                )

                timings["stt_ms"] = elapsed_ms(
                    stt_started_at
                )

                logger.info(
                    "TURN stt_complete request=%s "
                    "elapsed_ms=%s transcript=%r",
                    request_id,
                    timings["stt_ms"],
                    transcript,
                )

                yield stream_event({
                    "type": "transcript",
                    "request_id": request_id,
                    "text": transcript,
                    "timings": {
                        "stt_ms": timings["stt_ms"],
                    },
                })

                # -----------------------------------------
                # Build contextual LLM input
                # -----------------------------------------

                llm_messages = [
                    *session.messages,
                    {
                        "role": "user",
                        "content": transcript,
                    },
                ]

                sentence_queue: asyncio.Queue = (
                    asyncio.Queue()
                )

                llm_chunks: list[str] = []

                llm_error = None
                first_chunk_seen = False

                llm_started_at = (
                    time.perf_counter()
                )

                # -----------------------------------------
                # LLM producer
                # -----------------------------------------

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
                            if not first_chunk_seen:
                                first_chunk_seen = True

                                timings[
                                    "llm_first_chunk_ms"
                                ] = elapsed_ms(
                                    llm_started_at
                                )

                                logger.info(
                                    "TURN llm_first_chunk "
                                    "request=%s elapsed_ms=%s",
                                    request_id,
                                    timings[
                                        "llm_first_chunk_ms"
                                    ],
                                )

                            llm_chunks.append(chunk)

                            await sentence_queue.put(
                                chunk
                            )

                    except Exception as exc:
                        llm_error = exc

                    finally:
                        timings[
                            "llm_total_ms"
                        ] = elapsed_ms(
                            llm_started_at
                        )

                        await sentence_queue.put(
                            None
                        )

                producer_task = (
                    asyncio.create_task(
                        produce_llm()
                    )
                )

                # -----------------------------------------
                # Initial LLM buffer
                # -----------------------------------------

                initial_chunks: list[str] = []

                while (
                    len(initial_chunks)
                    < LLM_INITIAL_BUFFER_CHUNKS
                ):
                    item = await sentence_queue.get()

                    if item is None:
                        break

                    initial_chunks.append(item)

                logger.info(
                    "TURN initial_buffer_ready "
                    "request=%s chunks=%s "
                    "elapsed_ms=%s",
                    request_id,
                    len(initial_chunks),
                    elapsed_ms(total_started_at),
                )

                # -----------------------------------------
                # TTS streaming
                # -----------------------------------------

                audio_chunk_index = 0
                tts_total_ms = 0
                first_audio_ready = False

                async def synthesize_and_stream(
                    text_chunk: str,
                ):
                    """
                    Sends one LLM text chunk through CosyVoice.

                    CosyVoice may emit multiple audio chunks
                    before synthesis of the text chunk finishes.
                    """

                    nonlocal audio_chunk_index
                    nonlocal tts_total_ms
                    nonlocal first_audio_ready

                    tts_started_at = (
                        time.perf_counter()
                    )

                    logger.info(
                        "TURN tts_text_chunk_started "
                        "request=%s text=%r",
                        request_id,
                        text_chunk,
                    )

                    async for audio_response in (
                        model_manager.tts.stream_audio(
                            text_chunk
                        )
                    ):
                        audio_chunk_index += 1

                        current_tts_ms = elapsed_ms(
                            tts_started_at
                        )

                        if not first_audio_ready:
                            first_audio_ready = True

                            timings[
                                "first_audio_ready_ms"
                            ] = elapsed_ms(
                                total_started_at
                            )

                            logger.info(
                                "TURN first_audio_ready "
                                "request=%s elapsed_ms=%s",
                                request_id,
                                timings[
                                    "first_audio_ready_ms"
                                ],
                            )

                        logger.info(
                            "TURN audio_chunk_ready "
                            "request=%s chunk=%s "
                            "tts_elapsed_ms=%s "
                            "audio_bytes=%s",
                            request_id,
                            audio_chunk_index,
                            current_tts_ms,
                            len(audio_response),
                        )

                        yield stream_event({
                            "type": "audio",
                            "request_id": request_id,
                            "index": audio_chunk_index,
                            "text": text_chunk,
                            "audio_mime_type": (
                                "audio/wav"
                            ),
                            "audio_base64": (
                                base64.b64encode(
                                    audio_response
                                ).decode("ascii")
                            ),
                            "timings": {
                                "tts_elapsed_ms": (
                                    current_tts_ms
                                ),
                                "backend_elapsed_ms": (
                                    elapsed_ms(
                                        total_started_at
                                    )
                                ),
                            },
                        })

                    chunk_total_ms = elapsed_ms(
                        tts_started_at
                    )

                    tts_total_ms += chunk_total_ms

                    logger.info(
                        "TURN tts_text_chunk_complete "
                        "request=%s elapsed_ms=%s "
                        "text=%r",
                        request_id,
                        chunk_total_ms,
                        text_chunk,
                    )

                # -----------------------------------------
                # Process initial buffered chunks
                # -----------------------------------------

                for chunk in initial_chunks:
                    async for event in (
                        synthesize_and_stream(
                            chunk
                        )
                    ):
                        yield event

                # -----------------------------------------
                # Continue consuming LLM chunks
                # -----------------------------------------

                while True:
                    item = await sentence_queue.get()

                    if item is None:
                        break

                    async for event in (
                        synthesize_and_stream(
                            item
                        )
                    ):
                        yield event

                await producer_task

                if llm_error:
                    raise llm_error

                # -----------------------------------------
                # Finalize conversation context
                # -----------------------------------------

                response_text = " ".join(
                    llm_chunks
                ).strip()

                if not response_text:
                    raise RuntimeError(
                        "LLM returned an empty response"
                    )

                session_store.append_turn(
                    session=session,
                    user_text=transcript,
                    assistant_text=response_text,
                )

                timings[
                    "tts_total_ms"
                ] = tts_total_ms

                timings[
                    "backend_stream_complete_ms"
                ] = elapsed_ms(
                    total_started_at
                )

                logger.info(
                    "TURN complete request=%s "
                    "session=%s turn=%s "
                    "stt_ms=%s "
                    "llm_first_chunk_ms=%s "
                    "llm_total_ms=%s "
                    "tts_total_ms=%s "
                    "first_audio_ready_ms=%s "
                    "backend_total_ms=%s",
                    request_id,
                    session.session_id,
                    session.turn_count,
                    timings.get("stt_ms"),
                    timings.get(
                        "llm_first_chunk_ms"
                    ),
                    timings.get(
                        "llm_total_ms"
                    ),
                    timings.get(
                        "tts_total_ms"
                    ),
                    timings.get(
                        "first_audio_ready_ms"
                    ),
                    timings[
                        "backend_stream_complete_ms"
                    ],
                )

                yield stream_event({
                    "type": "complete",
                    "request_id": request_id,
                    "session_id": (
                        session.session_id
                    ),
                    "turn": session.turn_count,
                    "response_text": response_text,
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