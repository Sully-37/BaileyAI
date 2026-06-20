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