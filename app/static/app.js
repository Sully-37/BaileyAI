let sessionId = null;
let mediaRecorder = null;
let microphoneStream = null;
let recordingChunks = [];
let isRecording = false;
let activeAudio = null;

const modalEl = document.getElementById("termsModal");
const appContainerEl = document.getElementById("appContainer");

const statusEl = document.getElementById("status");
const instructionEl = document.getElementById("instruction");
const microphoneButtonEl = document.getElementById("microphoneButton");
const microphoneIconEl = document.getElementById("microphoneIcon");
const resetConversationEl = document.getElementById("resetConversation");
const pulseContainerEl = document.getElementById("pulseContainer");

const transcriptEl = document.getElementById("transcript");
const assistantEl = document.getElementById("assistant");


function setUiState(state, message = "") {
    document.body.dataset.state = state;

    pulseContainerEl.classList.add("hidden");
    microphoneButtonEl.classList.remove("recording", "processing", "speaking");

    switch (state) {
        case "loading":
            statusEl.textContent = "Preparing Bailey";
            instructionEl.textContent = message || "Loading AI models...";
            microphoneIconEl.textContent = "⏳";
            microphoneButtonEl.disabled = true;
            resetConversationEl.disabled = true;
            pulseContainerEl.classList.remove("hidden");
            break;

        case "ready":
            statusEl.textContent = "Bailey is ready";
            instructionEl.textContent = "Tap the microphone to talk";
            microphoneIconEl.textContent = "🎤";
            microphoneButtonEl.disabled = false;
            resetConversationEl.disabled = false;
            break;

        case "listening":
            statusEl.textContent = "Listening";
            instructionEl.textContent = "Tap again when you are finished";
            microphoneIconEl.textContent = "■";
            microphoneButtonEl.classList.add("recording");
            microphoneButtonEl.disabled = false;
            break;

        case "understanding":
            statusEl.textContent = "Understanding";
            instructionEl.textContent = "Bailey is reading your input";
            microphoneIconEl.textContent = "•••";
            microphoneButtonEl.classList.add("processing");
            microphoneButtonEl.disabled = true;
            pulseContainerEl.classList.remove("hidden");
            break;

        case "responding":
            statusEl.textContent = "Bailey is responding";
            instructionEl.textContent = "Audio is about to play";
            microphoneIconEl.textContent = "🔊";
            microphoneButtonEl.classList.add("speaking");
            microphoneButtonEl.disabled = true;
            pulseContainerEl.classList.remove("hidden");
            break;

        case "speaking":
            statusEl.textContent = "Bailey is speaking";
            instructionEl.textContent = "Listen for Bailey's response";
            microphoneIconEl.textContent = "🔊";
            microphoneButtonEl.classList.add("speaking");
            microphoneButtonEl.disabled = true;
            break;

        case "error":
            statusEl.textContent = "Bailey encountered a problem";
            instructionEl.textContent = message || "Please try again";
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
        throw new Error(`Request failed with HTTP ${response.status}`);
    }

    if (!response.ok) {
        const detail = data.detail || data;
        const message =
            detail.message ||
            detail.error ||
            `Request failed with HTTP ${response.status}`;

        throw new Error(message);
    }

    return data;
}


async function loadModels() {
    setUiState("loading", "Loading Bailey into GPU memory...");

    const response = await fetch("/startup/load-models", {
        method: "POST",
    });

    return readJsonResponse(response);
}


async function createSession() {
    const response = await fetch("/bailey/session", {
        method: "POST",
    });

    const data = await readJsonResponse(response);
    sessionId = data.session_id;
}


async function initializeBailey() {
    try {
        await loadModels();
        await createSession();
        setUiState("ready");

    } catch (error) {
        console.error("Bailey initialization failed:", error);
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

    microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
        },
    });

    const preferredMimeType = "audio/webm;codecs=opus";
    const options = MediaRecorder.isTypeSupported(preferredMimeType)
        ? { mimeType: preferredMimeType }
        : {};

    mediaRecorder = new MediaRecorder(
        microphoneStream,
        options,
    );

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordingChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        stopMicrophoneTracks();

        const mimeType =
            mediaRecorder.mimeType ||
            "audio/webm";

        const audioBlob = new Blob(
            recordingChunks,
            { type: mimeType },
        );

        await sendConversationTurn(audioBlob);
    };

    mediaRecorder.start();
    isRecording = true;
    setUiState("listening");
}


function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        return;
    }

    isRecording = false;
    setUiState("understanding");
    mediaRecorder.stop();
}


function stopMicrophoneTracks() {
    if (!microphoneStream) {
        return;
    }

    microphoneStream
        .getTracks()
        .forEach((track) => track.stop());

    microphoneStream = null;
}


async function sendConversationTurn(audioBlob) {
    try {
        if (!audioBlob.size) {
            throw new Error("The microphone recording was empty");
        }

        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append(
            "audio",
            audioBlob,
            "bailey-turn.webm",
        );

        const response = await fetch("/bailey/turn", {
            method: "POST",
            body: formData,
        });

        const data = await readJsonResponse(response);

        transcriptEl.textContent = `You: ${data.transcript}`;
        assistantEl.textContent = `Bailey: ${data.response_text}`;

        console.table(data.timings);

        setUiState("responding");

        await playBaileyAudio(
            data.audio_base64,
            data.audio_mime_type,
        );

    } catch (error) {
        console.error("Bailey turn failed:", error);

        if (error.message.toLowerCase().includes("session")) {
            sessionId = null;
        }

        setUiState("error", error.message);
    }
}


async function playBaileyAudio(
    audioBase64,
    mimeType = "audio/wav",
) {
    const binaryString = atob(audioBase64);
    const bytes = new Uint8Array(binaryString.length);

    for (let index = 0; index < binaryString.length; index += 1) {
        bytes[index] = binaryString.charCodeAt(index);
    }

    const audioBlob = new Blob(
        [bytes],
        { type: mimeType },
    );

    const audioUrl = URL.createObjectURL(audioBlob);

    activeAudio = new Audio(audioUrl);

    activeAudio.onplaying = () => {
        setUiState("speaking");
    };

    activeAudio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        activeAudio = null;
        setUiState("ready");
    };

    activeAudio.onerror = () => {
        URL.revokeObjectURL(audioUrl);
        activeAudio = null;
        setUiState("error", "The browser could not play Bailey's audio");
    };

    await activeAudio.play();
}


async function resetConversation() {
    try {
        if (sessionId) {
            await fetch(`/bailey/session/${sessionId}`, {
                method: "DELETE",
            });
        }

        sessionId = null;
        transcriptEl.textContent = "";
        assistantEl.textContent = "";

        await createSession();
        setUiState("ready");

    } catch (error) {
        console.error("Session reset failed:", error);
        setUiState("error", error.message);
    }
}


document.getElementById("acceptTerms").onclick = async () => {
    modalEl.style.display = "none";
    appContainerEl.classList.remove("hidden");
    await initializeBailey();
};


microphoneButtonEl.onclick = async () => {
    try {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    } catch (error) {
        stopMicrophoneTracks();
        isRecording = false;

        console.error("Microphone operation failed:", error);
        setUiState("error", error.message);
    }
};


resetConversationEl.onclick = resetConversation;


window.addEventListener("beforeunload", () => {
    stopMicrophoneTracks();

    if (sessionId) {
        navigator.sendBeacon(
            `/bailey/session/${sessionId}`,
        );
    }
});