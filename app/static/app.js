let sessionId = null;
let mediaRecorder = null;
let microphoneStream = null;
let recordingChunks = [];
let isRecording = false;
let isStopping = false;
let activeAudio = null;

let audioContext = null;
let analyser = null;
let microphoneSource = null;
let vadFrameId = null;

let recordingStartedAt = null;
let speechStartedAt = null;
let lastSpeechAt = null;
let speechDetected = false;

// Initial tuning values.
const SPEECH_RMS_THRESHOLD = 0.018;
const SILENCE_DURATION_MS = 650;
const MIN_SPEECH_DURATION_MS = 250;
const MAX_RECORDING_DURATION_MS = 30000;

const modalEl = document.getElementById("termsModal");
const appContainerEl = document.getElementById("appContainer");

const statusEl = document.getElementById("status");
const instructionEl = document.getElementById("instruction");
const microphoneButtonEl =
    document.getElementById("microphoneButton");
const microphoneIconEl =
    document.getElementById("microphoneIcon");
const resetConversationEl =
    document.getElementById("resetConversation");
const pulseContainerEl =
    document.getElementById("pulseContainer");

const transcriptEl = document.getElementById("transcript");
const assistantEl = document.getElementById("assistant");


function logLatency(event, details = {}) {
    console.info(
        `[Bailey][${new Date().toISOString()}] ${event}`,
        details,
    );
}


function setUiState(state, message = "") {
    document.body.dataset.state = state;

    pulseContainerEl.classList.add("hidden");

    microphoneButtonEl.classList.remove(
        "recording",
        "processing",
        "speaking",
    );

    switch (state) {
        case "loading":
            statusEl.textContent = "Preparing Bailey";
            instructionEl.textContent =
                message || "Loading AI models...";
            microphoneIconEl.textContent = "⏳";
            microphoneButtonEl.disabled = true;
            resetConversationEl.disabled = true;
            pulseContainerEl.classList.remove("hidden");
            break;

        case "ready":
            statusEl.textContent = "Bailey is ready";
            instructionEl.textContent =
                "Tap the microphone and speak naturally";
            microphoneIconEl.textContent = "🎤";
            microphoneButtonEl.disabled = false;
            resetConversationEl.disabled = false;
            break;

        case "listening":
            statusEl.textContent = speechDetected
                ? "Listening"
                : "Waiting for speech";

            instructionEl.textContent =
                "Bailey will detect when you stop speaking";

            microphoneIconEl.textContent = "🎤";

            microphoneButtonEl.classList.add(
                "recording"
            );

            microphoneButtonEl.disabled = false;
            break;

        case "understanding":
            statusEl.textContent = "Understanding";
            instructionEl.textContent =
                "Bailey is processing your voice";
            microphoneIconEl.textContent = "•••";

            microphoneButtonEl.classList.add(
                "processing"
            );

            microphoneButtonEl.disabled = true;
            pulseContainerEl.classList.remove("hidden");
            break;

        case "responding":
            statusEl.textContent = "Bailey is responding";
            instructionEl.textContent =
                "Audio is about to play";
            microphoneIconEl.textContent = "🔊";

            microphoneButtonEl.classList.add(
                "speaking"
            );

            microphoneButtonEl.disabled = true;
            pulseContainerEl.classList.remove("hidden");
            break;

        case "speaking":
            statusEl.textContent = "Bailey is speaking";
            instructionEl.textContent =
                "Listen for Bailey's response";
            microphoneIconEl.textContent = "🔊";

            microphoneButtonEl.classList.add(
                "speaking"
            );

            microphoneButtonEl.disabled = true;
            break;

        case "error":
            statusEl.textContent =
                "Bailey encountered a problem";

            instructionEl.textContent =
                message || "Please try again";

            microphoneIconEl.textContent = "!";
            microphoneButtonEl.disabled = false;
            resetConversationEl.disabled = false;
            break;
    }
}


async function readJsonResponse(response) {
    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(
            `Request failed with HTTP ${response.status}`
        );
    }

    if (!response.ok) {
        const detail = data.detail || data;

        throw new Error(
            detail.message ||
            detail.error ||
            `Request failed with HTTP ${response.status}`
        );
    }

    return data;
}


async function loadModels() {
    setUiState("loading", "Starting model load...");

    const response = await fetch(
        "/startup/load-models",
        {
            method: "POST",
        },
    );

    await readJsonResponse(response);
    await waitForModels();
}


async function waitForModels() {
    const pollIntervalMs = 3000;
    const timeoutMs = 15 * 60 * 1000;
    const startedAt = Date.now();

    while (Date.now() - startedAt < timeoutMs) {
        const response = await fetch("/health");
        const data = await readJsonResponse(response);
        const models = data.models;

        if (models.loaded) {
            return;
        }

        if (models.last_error) {
            throw new Error(models.last_error);
        }

        setUiState(
            "loading",
            `Loading models: STT ${models.stt ? "✓" : "..."}, ` +
            `LLM ${models.llm ? "✓" : "..."}, ` +
            `TTS ${models.tts ? "✓" : "..."}`
        );

        await new Promise(
            resolve => setTimeout(
                resolve,
                pollIntervalMs,
            )
        );
    }

    throw new Error("Model loading timed out");
}


async function createSession() {
    const response = await fetch(
        "/bailey/session",
        {
            method: "POST",
        },
    );

    const data = await readJsonResponse(response);

    sessionId = data.session_id;

    logLatency("session_created", {
        sessionId,
    });
}


async function initializeBailey() {
    try {
        await loadModels();
        await createSession();
        setUiState("ready");

    } catch (error) {
        console.error(
            "Bailey initialization failed:",
            error,
        );

        setUiState("error", error.message);
    }
}


async function startRecording() {
    if (!sessionId) {
        await createSession();
    }

    if (activeAudio) {
        activeAudio.pause();
        activeAudio = null;
    }

    transcriptEl.textContent = "";
    assistantEl.textContent = "";

    recordingChunks = [];
    isStopping = false;
    speechDetected = false;
    speechStartedAt = null;
    lastSpeechAt = null;
    recordingStartedAt = performance.now();

    microphoneStream =
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

    mediaRecorder = new MediaRecorder(
        microphoneStream,
        options,
    );

    mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            recordingChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        const captureEndedAt = performance.now();

        await stopVoiceDetection();
        stopMicrophoneTracks();

        const mimeType =
            mediaRecorder.mimeType || "audio/webm";

        const audioBlob = new Blob(
            recordingChunks,
            {
                type: mimeType,
            },
        );

        logLatency("audio_capture_complete", {
            capture_ms: Math.round(
                captureEndedAt - recordingStartedAt
            ),
            bytes: audioBlob.size,
            stop_reason: "silence_or_manual",
        });

        await sendConversationTurn(
            audioBlob,
            captureEndedAt,
        );
    };

    mediaRecorder.start(100);

    isRecording = true;

    await startVoiceDetection();
    setUiState("listening");

    logLatency("audio_capture_started", {
        silence_duration_ms: SILENCE_DURATION_MS,
        rms_threshold: SPEECH_RMS_THRESHOLD,
    });
}


async function startVoiceDetection() {
    audioContext = new AudioContext();

    if (audioContext.state === "suspended") {
        await audioContext.resume();
    }

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.2;

    microphoneSource =
        audioContext.createMediaStreamSource(
            microphoneStream
        );

    microphoneSource.connect(analyser);

    const samples =
        new Float32Array(analyser.fftSize);

    function inspectAudio() {
        if (!isRecording || isStopping) {
            return;
        }

        analyser.getFloatTimeDomainData(samples);

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
            if (!speechDetected) {
                speechDetected = true;
                speechStartedAt = now;

                logLatency("speech_detected", {
                    rms,
                    delay_from_record_start_ms:
                        Math.round(
                            now - recordingStartedAt
                        ),
                });
            }

            lastSpeechAt = now;
            setUiState("listening");
        }

        if (
            speechDetected &&
            speechStartedAt !== null &&
            lastSpeechAt !== null
        ) {
            const speechDuration =
                now - speechStartedAt;

            const silenceDuration =
                now - lastSpeechAt;

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
            now - recordingStartedAt >=
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

        vadFrameId =
            requestAnimationFrame(
                inspectAudio
            );
    }

    vadFrameId =
        requestAnimationFrame(
            inspectAudio
        );
}


async function stopVoiceDetection() {
    if (vadFrameId !== null) {
        cancelAnimationFrame(vadFrameId);
        vadFrameId = null;
    }

    if (microphoneSource) {
        microphoneSource.disconnect();
        microphoneSource = null;
    }

    analyser = null;

    if (audioContext) {
        await audioContext.close();
        audioContext = null;
    }
}


function stopRecording(reason = "manual") {
    if (
        !mediaRecorder ||
        mediaRecorder.state === "inactive" ||
        isStopping
    ) {
        return;
    }

    isStopping = true;
    isRecording = false;

    logLatency("recording_stopped", {
        reason,
    });

    setUiState("understanding");

    mediaRecorder.stop();
}


function stopMicrophoneTracks() {
    if (!microphoneStream) {
        return;
    }

    microphoneStream
        .getTracks()
        .forEach(track => track.stop());

    microphoneStream = null;
}


async function sendConversationTurn(
    audioBlob,
    captureEndedAt,
) {
    const requestStartedAt = performance.now();

    try {
        if (!audioBlob.size) {
            throw new Error(
                "The microphone recording was empty"
            );
        }

        const formData = new FormData();

        formData.append(
            "session_id",
            sessionId,
        );

        formData.append(
            "audio",
            audioBlob,
            "bailey-turn.webm",
        );

        logLatency("turn_request_started", {
            audio_bytes: audioBlob.size,
        });

        const response = await fetch(
            "/bailey/turn",
            {
                method: "POST",
                body: formData,
            },
        );

        const responseReceivedAt =
            performance.now();

        const data =
            await readJsonResponse(response);

        const jsonDecodedAt =
            performance.now();

        transcriptEl.textContent =
            `You: ${data.transcript}`;

        assistantEl.textContent =
            `Bailey: ${data.response_text}`;

        const browserTimings = {
            capture_to_request_ms: Math.round(
                requestStartedAt -
                captureEndedAt
            ),
            request_round_trip_ms: Math.round(
                responseReceivedAt -
                requestStartedAt
            ),
            response_decode_ms: Math.round(
                jsonDecodedAt -
                responseReceivedAt
            ),
        };

        console.table({
            ...data.timings,
            ...browserTimings,
        });

        logLatency("turn_response_received", {
            request_id: data.request_id,
            backend: data.timings,
            browser: browserTimings,
        });

        setUiState("responding");

        await playBaileyAudio(
            data.audio_base64,
            data.audio_mime_type,
            requestStartedAt,
        );

    } catch (error) {
        console.error(
            "Bailey turn failed:",
            error,
        );

        if (
            error.message
                .toLowerCase()
                .includes("session")
        ) {
            sessionId = null;
        }

        setUiState(
            "error",
            error.message,
        );
    }
}


async function playBaileyAudio(
    audioBase64,
    mimeType = "audio/wav",
    turnStartedAt,
) {
    const decodeStartedAt =
        performance.now();

    const binaryString =
        atob(audioBase64);

    const bytes =
        new Uint8Array(
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
            type: mimeType,
        },
    );

    const audioUrl =
        URL.createObjectURL(audioBlob);

    activeAudio = new Audio(audioUrl);

    activeAudio.onplaying = () => {
        const playbackStartedAt =
            performance.now();

        setUiState("speaking");

        logLatency("audio_playback_started", {
            audio_decode_ms: Math.round(
                playbackStartedAt -
                decodeStartedAt
            ),
            turn_to_audio_ms: Math.round(
                playbackStartedAt -
                turnStartedAt
            ),
        });
    };

    activeAudio.onended = () => {
        URL.revokeObjectURL(audioUrl);

        activeAudio = null;
        setUiState("ready");

        logLatency("audio_playback_complete", {
            total_turn_ms: Math.round(
                performance.now() -
                turnStartedAt
            ),
        });
    };

    activeAudio.onerror = () => {
        URL.revokeObjectURL(audioUrl);

        activeAudio = null;

        setUiState(
            "error",
            "The browser could not play Bailey's audio",
        );
    };

    await activeAudio.play();
}


async function resetConversation() {
    try {
        if (sessionId) {
            await fetch(
                `/bailey/session/${sessionId}`,
                {
                    method: "DELETE",
                },
            );
        }

        sessionId = null;
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


document
    .getElementById("acceptTerms")
    .onclick = async () => {
        modalEl.style.display = "none";

        appContainerEl.classList.remove(
            "hidden"
        );

        await initializeBailey();
    };


microphoneButtonEl.onclick = async () => {
    try {
        if (isRecording) {
            stopRecording("manual");
        } else {
            await startRecording();
        }

    } catch (error) {
        await stopVoiceDetection();
        stopMicrophoneTracks();

        isRecording = false;
        isStopping = false;

        console.error(
            "Microphone operation failed:",
            error,
        );

        setUiState(
            "error",
            error.message,
        );
    }
};


resetConversationEl.onclick =
    resetConversation;


window.addEventListener(
    "beforeunload",
    () => {
        stopMicrophoneTracks();
    },
);