import { state } from "./state.js";
import { setUiState } from "./ui.js";
import { logLatency, readJsonResponse } from "./utils.js";


export async function loadModels() {
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


export async function waitForModels() {
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


export async function createSession() {
    const response = await fetch(
        "/bailey/session",
        {
            method: "POST",
        },
    );

    const data = await readJsonResponse(response);

    state.sessionId = data.session_id;

    logLatency("session_created", {
        sessionId: state.sessionId,
    });
}


export async function initializeBailey() {
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
