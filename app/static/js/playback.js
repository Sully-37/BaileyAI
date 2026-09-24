
import { state } from "./state.js";
import { setUiState } from "./ui.js";
import { logLatency } from "./utils.js";

let audioContext = null;
let activeSources = new Set();

let streamComplete = false;
let nextStartTime = 0;
let pendingChunks = 0;
let generation = 0;

let turnStartedAt = null;
let firstPlaybackLogged = false;
let lastChunkIndex = null;

let playbackPromise = null;
let resolvePlayback = null;
let rejectPlayback = null;

/**
 * Initializes or resumes the browser audio engine.
 *
 * Call this directly from a user click so that browser
 * autoplay restrictions do not block Bailey's speech.
 */
export function unlockBaileyAudio() {
    const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;

    if (!AudioContextClass) {
        throw new Error(
            "This browser does not support Web Audio"
        );
    }

    if (
        !audioContext ||
        audioContext.state === "closed"
    ) {
        audioContext = new AudioContextClass({
            latencyHint: "interactive",
        });
    }

    if (audioContext.state === "suspended") {
        // Start resuming during the user gesture.
        // The returned promise is handled to avoid
        // an unhandled rejection.
        void audioContext.resume().catch(
            (error) => {
                console.warn(
                    "Audio unlock failed:",
                    error,
                );
            },
        );
    }
}

/**
 * Begins a new response and resets scheduling state.
 */
export function beginBaileyPlayback(
    requestStartedAt,
) {
    cancelBaileyPlayback();

    generation += 1;

    streamComplete = false;
    nextStartTime = 0;
    pendingChunks = 0;

    turnStartedAt = requestStartedAt;
    firstPlaybackLogged = false;
    lastChunkIndex = null;

    playbackPromise = new Promise(
        (resolve, reject) => {
            resolvePlayback = resolve;
            rejectPlayback = reject;
        },
    );

    // A backend error can arrive before conversation.js
    // reaches its eventual await of playbackPromise.
    // Attach a handler now to prevent an unhandled
    // rejection while preserving rejection for awaiters.
    void playbackPromise.catch(() => {});

    return playbackPromise;
}

/**
 * Converts base64-encoded, mono, signed PCM16 LE
 * into normalized Float32 samples.
 */
function decodePcm16(audioBase64) {
    const binary = atob(audioBase64);

    if (
        binary.length === 0 ||
        binary.length % 2 !== 0
    ) {
        throw new Error(
            "Bailey returned an invalid PCM audio packet"
        );
    }

    const samples = new Float32Array(
        binary.length / 2,
    );

    for (
        let index = 0;
        index < samples.length;
        index += 1
    ) {
        const byteIndex = index * 2;

        const unsigned =
            binary.charCodeAt(byteIndex) |
            (
                binary.charCodeAt(byteIndex + 1)
                << 8
            );

        const signed =
            unsigned >= 0x8000
                ? unsigned - 0x10000
                : unsigned;

        samples[index] = signed / 32768;
    }

    return samples;
}

/**
 * Schedules a PCM packet immediately after the
 * preceding packet in the AudioContext timeline.
 */
export function enqueueBaileyAudio(
    audioBase64,
    mimeType = "audio/pcm",
    chunkIndex = null,
    sampleRate = null,
) {
    if (!playbackPromise) {
        throw new Error(
            "Audio received without an active Bailey turn"
        );
    }

    if (streamComplete) {
        throw new Error(
            "Audio received after Bailey's stream completed"
        );
    }

    if (mimeType !== "audio/pcm") {
        throw new Error(
            `Unsupported Bailey audio format: ${mimeType}`
        );
    }

    if (
        !Number.isInteger(sampleRate) ||
        sampleRate <= 0
    ) {
        throw new Error(
            "Bailey did not provide a valid audio sample rate"
        );
    }

    unlockBaileyAudio();

    const currentGeneration = generation;
    const decodeStartedAt = performance.now();

    const samples = decodePcm16(
        audioBase64,
    );

    const audioBuffer =
        audioContext.createBuffer(
            1,
            samples.length,
            sampleRate,
        );

    audioBuffer.copyToChannel(
        samples,
        0,
    );

    const source =
        audioContext.createBufferSource();

    source.buffer = audioBuffer;
    source.connect(
        audioContext.destination,
    );

    const now = audioContext.currentTime;

    // A small initial lead gives the audio engine
    // time to accept the first scheduled packet.
    // Subsequent packets are placed consecutively.
    const scheduledStart = Math.max(
        nextStartTime,
        now + 0.02,
    );

    nextStartTime =
        scheduledStart +
        audioBuffer.duration;

    pendingChunks += 1;
    lastChunkIndex = chunkIndex;

    activeSources.add(source);
    state.activeAudio = source;

    logLatency(
        "audio_chunk_queued",
        {
            chunk_index: chunkIndex,
            pending_chunks: pendingChunks,
            audio_duration_ms: Math.round(
                audioBuffer.duration * 1000,
            ),
            scheduled_delay_ms: Math.round(
                (scheduledStart - now) * 1000,
            ),
        },
    );

    source.onended = () => {
        activeSources.delete(source);

        if (
            currentGeneration !== generation
        ) {
            return;
        }

        pendingChunks -= 1;

        if (state.activeAudio === source) {
            state.activeAudio = null;
        }

        logLatency(
            "audio_chunk_playback_complete",
            {
                chunk_index: chunkIndex,
                pending_chunks: pendingChunks,
            },
        );

        if (
            streamComplete &&
            pendingChunks === 0
        ) {
            finishPlayback();
        }
    };

    try {
        source.start(scheduledStart);
    } catch (error) {
        activeSources.delete(source);
        pendingChunks -= 1;
        source.disconnect();
        throw error;
    }

    if (!firstPlaybackLogged) {
        firstPlaybackLogged = true;

        setUiState("speaking");

        const estimatedPlaybackAt =
            performance.now() +
            Math.max(
                0,
                scheduledStart -
                audioContext.currentTime,
            ) * 1000;

        logLatency(
            "first_audio_playback_scheduled",
            {
                chunk_index: chunkIndex,
                request_to_audio_ms:
                    Math.round(
                        estimatedPlaybackAt -
                        turnStartedAt,
                    ),
                audio_decode_ms:
                    Math.round(
                        performance.now() -
                        decodeStartedAt,
                    ),
                audio_context_state:
                    audioContext.state,
            },
        );
    }
}

/**
 * Indicates that no more audio packets will arrive.
 * Playback completes after the final scheduled packet.
 */
export function markPlaybackStreamComplete() {
    streamComplete = true;

    if (pendingChunks === 0) {
        finishPlayback();
    }
}

/**
 * Stops scheduled audio and rejects the current turn.
 */
export function failPlayback(error) {
    const failure =
        error instanceof Error
            ? error
            : new Error(String(error));

    generation += 1;
    streamComplete = true;

    stopAllSources();

    pendingChunks = 0;
    nextStartTime = 0;

    if (rejectPlayback) {
        rejectPlayback(failure);
    }

    resetPromiseState();
}

/**
 * Stops any current playback when starting another
 * turn or resetting the conversation.
 */
export function cancelBaileyPlayback() {
    if (rejectPlayback) {
        rejectPlayback(
            new Error("Bailey playback was cancelled"),
        );
    }

    generation += 1;
    streamComplete = true;

    stopAllSources();

    pendingChunks = 0;
    nextStartTime = 0;

    resetPromiseState();
}

/**
 * Stops and disconnects every active or scheduled
 * Web Audio source.
 */
function stopAllSources() {
    for (const source of activeSources) {
        source.onended = null;

        try {
            source.stop();
        } catch {
            // The source may already have stopped.
        }

        source.disconnect();
    }

    activeSources.clear();
    state.activeAudio = null;
}

/**
 * Resolves the turn after the last audio packet
 * has actually finished playing.
 */
function finishPlayback() {
    if (!playbackPromise) {
        return;
    }

    logLatency(
        "audio_playback_complete",
        {
            total_turn_ms: Math.round(
                performance.now() -
                turnStartedAt,
            ),
            last_chunk_index:
                lastChunkIndex,
        },
    );

    setUiState("ready");

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