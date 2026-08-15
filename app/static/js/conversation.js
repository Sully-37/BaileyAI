import { state } from "./state.js";
import {
    transcriptEl,
    assistantEl,
} from "./dom.js";
import { createSession } from "./api.js";
import {
    beginBaileyPlayback,
    enqueueBaileyAudio,
    markPlaybackStreamComplete,
    failPlayback,
} from "./playback.js";
import { setUiState } from "./ui.js";
import { logLatency } from "./utils.js";


export async function sendConversationTurn(
    audioBlob,
    captureEndedAt,
) {
    const requestStartedAt =
        performance.now();

    let playbackStarted = false;
    let playbackPromise = null;

    try {
        if (!audioBlob.size) {
            throw new Error(
                "The microphone recording was empty"
            );
        }

        const formData = new FormData();

        formData.append(
            "session_id",
            state.sessionId,
        );

        formData.append(
            "audio",
            audioBlob,
            "bailey-turn.webm",
        );

        const captureToRequestMs =
            Math.round(
                requestStartedAt -
                captureEndedAt
            );

        logLatency(
            "turn_request_started",
            {
                audio_bytes:
                    audioBlob.size,
                capture_to_request_ms:
                    captureToRequestMs,
            },
        );

        playbackPromise =
            beginBaileyPlayback(
                requestStartedAt
            );

        playbackStarted = true;

        const response = await fetch(
            "/bailey/turn",
            {
                method: "POST",
                body: formData,
            },
        );

        if (!response.ok) {
            let message =
                `Request failed with HTTP ${response.status}`;

            try {
                const data =
                    await response.json();

                const detail =
                    data.detail || data;

                message =
                    detail.message ||
                    detail.error ||
                    message;

            } catch {
                // Preserve default message.
            }

            throw new Error(message);
        }

        if (!response.body) {
            throw new Error(
                "Streaming response body is unavailable"
            );
        }

        const reader =
            response.body.getReader();

        const decoder =
            new TextDecoder();

        let pendingText = "";
        let firstEventReceived = false;
        let firstAudioEventReceived = false;
        let streamCompleted = false;

        while (true) {
            const {
                done,
                value,
            } = await reader.read();

            if (done) {
                break;
            }

            pendingText += decoder.decode(
                value,
                {
                    stream: true,
                },
            );

            let newlineIndex;

            while (
                (
                    newlineIndex =
                        pendingText.indexOf("\n")
                ) >= 0
            ) {
                const line =
                    pendingText
                        .slice(
                            0,
                            newlineIndex
                        )
                        .trim();

                pendingText =
                    pendingText.slice(
                        newlineIndex + 1
                    );

                if (!line) {
                    continue;
                }

                const event =
                    JSON.parse(line);

                if (!firstEventReceived) {
                    firstEventReceived = true;

                    logLatency(
                        "first_stream_event",
                        {
                            elapsed_ms:
                                Math.round(
                                    performance.now()
                                    -
                                    requestStartedAt
                                ),
                            type:
                                event.type,
                        },
                    );
                }

                if (
                    event.type ===
                    "transcript"
                ) {
                    transcriptEl.textContent =
                        `You: ${event.text}`;

                    continue;
                }

                if (
                    event.type ===
                    "audio"
                ) {
                    if (
                        !firstAudioEventReceived
                    ) {
                        firstAudioEventReceived =
                            true;

                        logLatency(
                            "first_audio_chunk_received",
                            {
                                elapsed_ms:
                                    Math.round(
                                        performance.now()
                                        -
                                        requestStartedAt
                                    ),
                                chunk_index:
                                    event.index,
                                backend:
                                    event.timings,
                            },
                        );
                    }

                    setUiState(
                        "responding"
                    );

                    enqueueBaileyAudio(
                        event.audio_base64,
                        event.audio_mime_type,
                        event.index,
                    );

                    continue;
                }

                if (
                    event.type ===
                    "complete"
                ) {
                    streamCompleted = true;

                    assistantEl.textContent =
                        `Bailey: ${event.response_text}`;

                    console.table(
                        event.timings
                    );

                    logLatency(
                        "backend_stream_complete",
                        {
                            request_id:
                                event.request_id,
                            backend:
                                event.timings,
                            browser_elapsed_ms:
                                Math.round(
                                    performance.now()
                                    -
                                    requestStartedAt
                                ),
                        },
                    );

                    markPlaybackStreamComplete();

                    continue;
                }

                if (
                    event.type ===
                    "error"
                ) {
                    throw new Error(
                        event.message ||
                        "Bailey turn failed"
                    );
                }
            }
        }

        if (pendingText.trim()) {
            const event =
                JSON.parse(
                    pendingText.trim()
                );

            if (
                event.type === "error"
            ) {
                throw new Error(
                    event.message
                );
            }
        }

        if (!streamCompleted) {
            throw new Error(
                "Bailey response stream ended unexpectedly"
            );
        }

        await playbackPromise;

    } catch (error) {
        console.error(
            "Bailey turn failed:",
            error,
        );

        if (playbackStarted) {
            failPlayback(error);
        }

        if (
            error.message
                .toLowerCase()
                .includes("session")
        ) {
            state.sessionId = null;
        }

        setUiState(
            "error",
            error.message,
        );
    }
}


export async function resetConversation() {
    try {
        if (state.sessionId) {
            await fetch(
                `/bailey/session/${state.sessionId}`,
                {
                    method: "DELETE",
                },
            );
        }

        state.sessionId = null;

        transcriptEl.textContent = "";
        assistantEl.textContent = "";

        await createSession();

        setUiState("ready");

    } catch (error) {
        console.error(
            "Session reset failed:",
            error,
        );

        setUiState(
            "error",
            error.message,
        );
    }
}