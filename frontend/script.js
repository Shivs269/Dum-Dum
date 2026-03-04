const video = document.getElementById("video");
const description = document.getElementById("description");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

/* ---------- CAMERA SETUP ---------- */

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Camera error:", err));


/* ---------- SYNC CANVAS WITH VIDEO ---------- */

function syncCanvasSize() {
    canvas.width = video.offsetWidth;
    canvas.height = video.offsetHeight;
}

video.addEventListener("loadedmetadata", syncCanvasSize);
window.addEventListener("resize", syncCanvasSize);


/* ---------- FRAME CAPTURE + ANALYSIS ---------- */

function captureAndAnalyze() {

    if (!video.videoWidth) return;

    // Capture real-resolution frame
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;

    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(video, 0, 0);

    const frameData = tempCanvas.toDataURL("image/jpeg");

    fetch("http://127.0.0.1:3000/analyze_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameData })
    })
    .then(res => res.json())
    .then(data => {

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!data.detections) return;

        // Scale detection boxes from raw resolution → displayed resolution
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        data.detections.forEach(det => {

            let [x1, y1, x2, y2] = det.box;

            x1 *= scaleX;
            x2 *= scaleX;
            y1 *= scaleY;
            y2 *= scaleY;

            let color = "green";
            if (det.color === "yellow") color = "yellow";
            if (det.color === "red") color = "red";

            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Draw label
            ctx.fillStyle = color;
            ctx.font = "16px Arial";
            ctx.fillText(
                `${det.object} (${det.confidence})`,
                x1,
                y1 > 20 ? y1 - 5 : y1 + 15
            );
        });

        // Threat text
        if (data.danger)
            description.innerText = "⚠️ Threat detected";
        else
            description.innerText = "✔️ Area safe";

    })
    .catch(console.error);
}


/* ---------- CONTINUOUS DETECTION LOOP ---------- */

// Runs every 1.5 seconds (safe + realistic for surveillance)
setInterval(captureAndAnalyze, 1500);
