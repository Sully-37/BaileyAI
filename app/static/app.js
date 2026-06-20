let ws;

// Browser media recorder for microphone streaming.
let mediaRecorder;

// Active microphone stream.
let stream;

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const assistantEl = document.getElementById("assistant");

const modalEl = document.getElementById("termsModal");
const appContainerEl = document.getElementById("appContainer");


// Enables application after terms acceptance.
document.getElementById("acceptTerms").onclick = () => {
    modalEl.style.display = "none";
    appContainerEl.classList.remove("hidden");
};


// Loads all inference runtimes into GPU VRAM.
document.getElementById("loadModels").onclick = async () => {
    const response = await fetch("/startup/load-models", {
        method: "POST",
    });

    const data = await response.json();

    statusEl.textContent = JSON.stringify(data);
};


// Opens websocket connection to realtime backend.
document.getElementById("connect").onclick = async () => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";

    ws = new WebSocket(`${protocol}://${window.location.host}/ws/voice`);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        statusEl.textContent = "Connected";
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === "string") {
            const msg = JSON.parse(event.data);

            if (msg.type === "transcript") {
                transcriptEl.textContent = msg.text;
            }

            if (msg.type === "assistant_text") {
                assistantEl.textContent += " " + msg.text;
            }
        } else {
            const blob = new Blob([event.data], { type: "audio/wav" });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.play();
        }
    };
};


// Starts microphone capture and streams chunks to backend.
document.getElementById("start").onclick = async () => {
    transcriptEl.textContent = "";
    assistantEl.textContent = "";

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm"
    });

    mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            const buffer = await event.data.arrayBuffer();
            ws.send(buffer);
        }
    };

    mediaRecorder.start(250);

    statusEl.textContent = "Recording";
};


// Stops recording and signals utterance completion.
document.getElementById("stop").onclick = async () => {
    mediaRecorder.stop();

    stream.getTracks().forEach(track => track.stop());

    ws.send("end_utterance");

    statusEl.textContent = "Processing";
};