
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
    cancelBaileyPlayback,
} from "./playback.js";
import { setUiState } from "./ui.js";
import { logLatency } from "./utils.js";

let activeTurnController = null;

export async function sendConversationTurn(
    audioBlob,
    captureEndedAt,
) {
    const requestStartedAt =
        performance.now();

    const controller =
        new AbortController();

    activeTurnController = controller;

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

        logLatency(
            "turn_request_started",
            {
                audio_bytes: audioBlob.size,
                capture_to_request_ms:
                    Math.round(
                        requestStartedAt -
                        captureEndedAt,
                    ),
            },
        );

        playbackPromise =
            beginBaileyPlayback(
                requestStartedAt,
            );

        playbackStarted = true;

        const response = await fetch(
            "/bailey/turn",
            {
                method: "POST",
                body: formData,
                signal: controller.signal,
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
                // Preserve the HTTP error message.
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

        /**
         * Handles one decoded NDJSON event.
         */
        function handleEvent(event) {
            if (!firstEventReceived) {
                firstEventReceived = true;

                logLatency(
                    "first_stream_event",
                    {
                        elapsed_ms:
                            Math.round(
                                performance.now() -
                                requestStartedAt,
                            ),
                        type: event.type,
                    },
                );
            }

            if (event.type === "transcript") {
                transcriptEl.textContent =
                    `You: ${event.text}`;

                return;
            }

            if (event.type === "audio") {
                if (!firstAudioEventReceived) {
                    firstAudioEventReceived = true;

                    logLatency(
                        "first_audio_chunk_received",
                        {
                            elapsed_ms:
                                Math.round(
                                    performance.now() -
                                    requestStartedAt,
                                ),
                            chunk_index:
                                event.index,
                            backend:
                                event.timings,
                        },
                    );
                }

                enqueueBaileyAudio(
                    event.audio_base64,
                    event.audio_mime_type,
                    event.index,
                    event.sample_rate,
                );

                return;
            }

            if (event.type === "complete") {
                streamCompleted = true;

                assistantEl.textContent =
                    `Bailey: ${event.response_text}`;

                console.table(
                    event.timings,
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
                                performance.now() -
                                requestStartedAt,
                            ),
                    },
                );

                markPlaybackStreamComplete();

                return;
            }

            if (event.type === "error") {
                throw new Error(
                    event.message ||
                    "Bailey turn failed",
                );
            }
        }

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
                            newlineIndex,
                        )
                        .trim();

                pendingText =
                    pendingText.slice(
                        newlineIndex + 1,
                    );

                if (!line) {
                    continue;
                }

                handleEvent(
                    JSON.parse(line),
                );
            }
        }

        pendingText += decoder.decode();

        if (pendingText.trim()) {
            handleEvent(
                JSON.parse(
                    pendingText.trim(),
                ),
            );
        }

        if (!streamCompleted) {
            throw new Error(
                "Bailey response stream ended unexpectedly"
            );
        }

        await playbackPromise;

    } catch (error) {
        if (controller.signal.aborted) {
            return;
        }

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

    } finally {
        if (activeTurnController === controller) {
            activeTurnController = null;
        }
    }
}

export async function resetConversation() {
    try {
        if (activeTurnController) {
            activeTurnController.abort();
            activeTurnController = null;
        }

        cancelBaileyPlayback();

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