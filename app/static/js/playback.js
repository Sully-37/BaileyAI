import { state } from "./state.js";
import { setUiState } from "./ui.js";
import { logLatency } from "./utils.js";


let audioQueue = [];
let isPlaying = false;
let streamComplete = false;

let turnStartedAt = null;
let firstPlaybackLogged = false;

let playbackPromise = null;
let resolvePlayback = null;
let rejectPlayback = null;


export function beginBaileyPlayback(
    requestStartedAt,
) {
    audioQueue = [];
    isPlaying = false;
    streamComplete = false;

    turnStartedAt = requestStartedAt;
    firstPlaybackLogged = false;

    playbackPromise = new Promise(
        (resolve, reject) => {
            resolvePlayback = resolve;
            rejectPlayback = reject;
        }
    );

    return playbackPromise;
}


export function enqueueBaileyAudio(
    audioBase64,
    mimeType = "audio/wav",
    chunkIndex = null,
) {
    audioQueue.push({
        audioBase64,
        mimeType,
        chunkIndex,
    });

    logLatency("audio_chunk_queued", {
        chunk_index: chunkIndex,
        queue_depth: audioQueue.length,
    });

    if (!isPlaying) {
        void playNextChunk();
    }
}


export function markPlaybackStreamComplete() {
    streamComplete = true;

    if (
        !isPlaying &&
        audioQueue.length === 0
    ) {
        finishPlayback();
    }
}


export function failPlayback(error) {
    audioQueue = [];
    streamComplete = true;

    if (state.activeAudio) {
        state.activeAudio.pause();
        state.activeAudio = null;
    }

    isPlaying = false;

    if (rejectPlayback) {
        rejectPlayback(error);
    }

    resetPromiseState();
}


async function playNextChunk() {
    if (audioQueue.length === 0) {
        isPlaying = false;

        if (streamComplete) {
            finishPlayback();
        }

        return;
    }

    isPlaying = true;

    const chunk = audioQueue.shift();

    const decodeStartedAt =
        performance.now();

    const binaryString =
        atob(chunk.audioBase64);

    const bytes = new Uint8Array(
        binaryString.length
    );

    for (
        let index = 0;
        index < binaryString.length;
        index += 1
    ) {
        bytes[index] =
            binaryString.charCodeAt(index);
    }

    const audioBlob = new Blob(
        [bytes],
        {
            type: chunk.mimeType,
        },
    );

    const audioUrl =
        URL.createObjectURL(audioBlob);

    state.activeAudio =
        new Audio(audioUrl);

    state.activeAudio.onplaying = () => {
        const now = performance.now();

        setUiState("speaking");

        if (!firstPlaybackLogged) {
            firstPlaybackLogged = true;

            logLatency(
                "first_audio_playback_started",
                {
                    chunk_index:
                        chunk.chunkIndex,
                    request_to_audio_ms:
                        Math.round(
                            now - turnStartedAt
                        ),
                    audio_decode_ms:
                        Math.round(
                            now -
                            decodeStartedAt
                        ),
                },
            );
        } else {
            logLatency(
                "audio_chunk_playback_started",
                {
                    chunk_index:
                        chunk.chunkIndex,
                    queue_depth:
                        audioQueue.length,
                },
            );
        }
    };

    state.activeAudio.onended = () => {
        URL.revokeObjectURL(audioUrl);

        state.activeAudio = null;
        isPlaying = false;

        logLatency(
            "audio_chunk_playback_complete",
            {
                chunk_index:
                    chunk.chunkIndex,
                queue_depth:
                    audioQueue.length,
            },
        );

        void playNextChunk();
    };

    state.activeAudio.onerror = () => {
        URL.revokeObjectURL(audioUrl);

        state.activeAudio = null;
        isPlaying = false;

        const error = new Error(
            "The browser could not play Bailey's audio"
        );

        setUiState(
            "error",
            error.message,
        );

        failPlayback(error);
    };

    try {
        await state.activeAudio.play();
    } catch (error) {
        failPlayback(error);
    }
}


function finishPlayback() {
    setUiState("ready");

    logLatency(
        "audio_playback_complete",
        {
            total_turn_ms: Math.round(
                performance.now()
                - turnStartedAt
            ),
        },
    );

    if (resolvePlayback) {
        resolvePlayback();
    }

    resetPromiseState();
}


function resetPromiseState() {
    playbackPromise = null;
    resolvePlayback = null;
    rejectPlayback = null;
}