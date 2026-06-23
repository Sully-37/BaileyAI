let ws;
let mediaRecorder;
let stream;

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const assistantEl = document.getElementById("assistant");
const missionOutputEl = document.getElementById("missionOutput");

const modalEl = document.getElementById("termsModal");
const appContainerEl = document.getElementById("appContainer");


// Enables application after terms acceptance.
document.getElementById("acceptTerms").onclick = () => {
    modalEl.style.display = "none";
    appContainerEl.classList.remove("hidden");
};


// Records a short browser mic sample and sends it to mission control.
document.getElementById("missionTest").onclick = async () => {
    statusEl.textContent = "Running mission control test...";
    missionOutputEl.textContent = "";

    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(micStream, { mimeType: "audio/webm" });

    const chunks = [];

    recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };

    recorder.onstop = async () => {
        micStream.getTracks().forEach(track => track.stop());

        const audioBlob = new Blob(chunks, { type: "audio/webm" });

        const formData = new FormData();
        formData.append("audio", audioBlob, "mission-test.webm");

        const response = await fetch("/mission-control/test", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        missionOutputEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = "Mission control test complete";
    };

    recorder.start();

    setTimeout(() => {
        recorder.stop();
    }, 1500);
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

            if (msg.type === "error") {
                statusEl.textContent = msg.message;
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