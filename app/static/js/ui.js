import { state } from "./state.js";
import {
    statusEl,
    instructionEl,
    microphoneButtonEl,
    microphoneIconEl,
    resetConversationEl,
    pulseContainerEl,
} from "./dom.js";


export function setUiState(stateName, message = "") {
    document.body.dataset.state = stateName;

    pulseContainerEl.classList.add("hidden");

    microphoneButtonEl.classList.remove(
        "recording",
        "processing",
        "speaking",
    );

    switch (stateName) {
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
            statusEl.textContent = state.speechDetected
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
