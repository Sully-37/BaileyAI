export const state = {
    sessionId: null,
    mediaRecorder: null,
    microphoneStream: null,
    recordingChunks: [],
    isRecording: false,
    isStopping: false,
    activeAudio: null,

    audioContext: null,
    analyser: null,
    microphoneSource: null,
    vadFrameId: null,

    recordingStartedAt: null,
    speechStartedAt: null,
    lastSpeechAt: null,
    speechDetected: false,
};

// Initial tuning values.
export const SPEECH_RMS_THRESHOLD = 0.018;
export const SILENCE_DURATION_MS = 650;
export const MIN_SPEECH_DURATION_MS = 250;
export const MAX_RECORDING_DURATION_MS = 30000;
