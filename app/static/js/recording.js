import {
    state,
    SPEECH_RMS_THRESHOLD,
    SILENCE_DURATION_MS,
    MIN_SPEECH_DURATION_MS,
    MAX_RECORDING_DURATION_MS,
} from "./state.js";
import { transcriptEl, assistantEl } from "./dom.js";
import { createSession } from "./api.js";
import { sendConversationTurn } from "./conversation.js";
import { setUiState } from "./ui.js";
import { logLatency } from "./utils.js";


export async function startRecording() {
    if (!state.sessionId) {
        await createSession();
    }

    if (state.activeAudio) {
        state.activeAudio.pause();
        state.activeAudio = null;
    }

    transcriptEl.textContent = "";
    assistantEl.textContent = "";

    state.recordingChunks = [];
    state.isStopping = false;
    state.speechDetected = false;
    state.speechStartedAt = null;
    state.lastSpeechAt = null;
    state.recordingStartedAt = performance.now();

    state.microphoneStream =
        await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

    const preferredMimeType =
        "audio/webm;codecs=opus";

    const options =
        MediaRecorder.isTypeSupported(
            preferredMimeType
        )
            ? { mimeType: preferredMimeType }
            : {};

    state.mediaRecorder = new MediaRecorder(
        state.microphoneStream,
        options,
    );

    state.mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            state.recordingChunks.push(event.data);
        }
    };

    state.mediaRecorder.onstop = async () => {
        const captureEndedAt = performance.now();

        await stopVoiceDetection();
        stopMicrophoneTracks();

        const mimeType =
            state.mediaRecorder.mimeType || "audio/webm";

        const audioBlob = new Blob(
            state.recordingChunks,
            {
                type: mimeType,
            },
        );

        logLatency("audio_capture_complete", {
            capture_ms: Math.round(
                captureEndedAt - state.recordingStartedAt
            ),
            bytes: audioBlob.size,
            stop_reason: "silence_or_manual",
        });

        await sendConversationTurn(
            audioBlob,
            captureEndedAt,
        );
    };

    state.mediaRecorder.start(100);

    state.isRecording = true;

    await startVoiceDetection();
    setUiState("listening");

    logLatency("audio_capture_started", {
        silence_duration_ms: SILENCE_DURATION_MS,
        rms_threshold: SPEECH_RMS_THRESHOLD,
    });
}


export async function startVoiceDetection() {
    state.audioContext = new AudioContext();

    if (state.audioContext.state === "suspended") {
        await state.audioContext.resume();
    }

    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 2048;
    state.analyser.smoothingTimeConstant = 0.2;

    state.microphoneSource =
        state.audioContext.createMediaStreamSource(
            state.microphoneStream
        );

    state.microphoneSource.connect(state.analyser);

    const samples =
        new Float32Array(state.analyser.fftSize);

    function inspectAudio() {
        if (!state.isRecording || state.isStopping) {
            return;
        }

        state.analyser.getFloatTimeDomainData(samples);

        let sumSquares = 0;

        for (
            let index = 0;
            index < samples.length;
            index += 1
        ) {
            sumSquares += samples[index] ** 2;
        }

        const rms = Math.sqrt(
            sumSquares / samples.length
        );

        const now = performance.now();

        if (rms >= SPEECH_RMS_THRESHOLD) {
            if (!state.speechDetected) {
                state.speechDetected = true;
                state.speechStartedAt = now;

                logLatency("speech_detected", {
                    rms,
                    delay_from_record_start_ms:
                        Math.round(
                            now - state.recordingStartedAt
                        ),
                });
            }

            state.lastSpeechAt = now;
            setUiState("listening");
        }

        if (
            state.speechDetected &&
            state.speechStartedAt !== null &&
            state.lastSpeechAt !== null
        ) {
            const speechDuration =
                now - state.speechStartedAt;

            const silenceDuration =
                now - state.lastSpeechAt;

            if (
                speechDuration >=
                    MIN_SPEECH_DURATION_MS &&
                silenceDuration >=
                    SILENCE_DURATION_MS
            ) {
                logLatency(
                    "end_of_speech_detected",
                    {
                        speech_duration_ms:
                            Math.round(
                                speechDuration
                            ),
                        silence_duration_ms:
                            Math.round(
                                silenceDuration
                            ),
                    },
                );

                stopRecording(
                    "silence_detected"
                );

                return;
            }
        }

        if (
            now - state.recordingStartedAt >=
            MAX_RECORDING_DURATION_MS
        ) {
            logLatency(
                "maximum_recording_reached"
            );

            stopRecording(
                "maximum_duration"
            );

            return;
        }

        state.vadFrameId =
            requestAnimationFrame(
                inspectAudio
            );
    }

    state.vadFrameId =
        requestAnimationFrame(
            inspectAudio
        );
}


export async function stopVoiceDetection() {
    if (state.vadFrameId !== null) {
        cancelAnimationFrame(state.vadFrameId);
        state.vadFrameId = null;
    }

    if (state.microphoneSource) {
        state.microphoneSource.disconnect();
        state.microphoneSource = null;
    }

    state.analyser = null;

    if (state.audioContext) {
        await state.audioContext.close();
        state.audioContext = null;
    }
}


export function stopRecording(reason = "manual") {
    if (
        !state.mediaRecorder ||
        state.mediaRecorder.state === "inactive" ||
        state.isStopping
    ) {
        return;
    }

    state.isStopping = true;
    state.isRecording = false;

    logLatency("recording_stopped", {
        reason,
    });

    setUiState("understanding");

    state.mediaRecorder.stop();
}


export function stopMicrophoneTracks() {
    if (!state.microphoneStream) {
        return;
    }

    state.microphoneStream
        .getTracks()
        .forEach(track => track.stop());

    state.microphoneStream = null;
}
