
import { state } from "./state.js";
import {
    modalEl,
    appContainerEl,
    microphoneButtonEl,
    resetConversationEl,
    acceptTermsEl,
} from "./dom.js";
import { initializeBailey } from "./api.js";
import { resetConversation } from "./conversation.js";
import {
    startRecording,
    stopRecording,
    stopVoiceDetection,
    stopMicrophoneTracks,
} from "./recording.js";
import {
    unlockBaileyAudio,
    cancelBaileyPlayback,
} from "./playback.js";
import { setUiState } from "./ui.js";

acceptTermsEl.onclick = async () => {
    // Initialize audio during a user gesture.
    unlockBaileyAudio();

    modalEl.style.display = "none";

    appContainerEl.classList.remove(
        "hidden",
    );

    await initializeBailey();
};

microphoneButtonEl.onclick = async () => {
    try {
        // Resume audio during the click if the
        // browser suspended it since initialization.
        unlockBaileyAudio();

        if (state.isRecording) {
            stopRecording("manual");
        } else {
            await startRecording();
        }

    } catch (error) {
        await stopVoiceDetection();
        stopMicrophoneTracks();

        state.isRecording = false;
        state.isStopping = false;

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
        cancelBaileyPlayback();
        stopMicrophoneTracks();
    },
);