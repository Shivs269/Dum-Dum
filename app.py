from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLO Model
model = YOLO("yolov8m.pt")

# Object categories
DANGEROUS = ["knife", "scissors", "gun", "pistol", "rifle", "sword"]
MODERATE = ["bottle", "backpack", "hammer"]

# -----------------------------
# 🔥 RAG Threat Knowledge Base
# -----------------------------
THREAT_KNOWLEDGE = {

    "knife": {
        "risk": "Used as a weapon capable of causing serious harm.",
        "context": "Knives are frequently restricted in public transport and secure areas.",
        "action": "Immediate security attention recommended."
    },

    "gun": {
        "risk": "Firearm capable of lethal force.",
        "context": "Unauthorized firearm presence is extremely dangerous in crowded areas.",
        "action": "Urgent law enforcement response required."
    },

    "pistol": {
        "risk": "Compact firearm capable of lethal force.",
        "context": "Often concealed and highly dangerous in public spaces.",
        "action": "Urgent law enforcement response required."
    },

    "rifle": {
        "risk": "Long range firearm capable of mass casualty events.",
        "context": "Highly restricted in civilian transport environments.",
        "action": "Immediate lockdown and security escalation required."
    },

    "backpack": {
        "risk": "Can conceal hazardous materials or suspicious items.",
        "context": "Unattended bags are common indicators of security threats.",
        "action": "Monitoring and inspection advised."
    },

    "bottle": {
        "risk": "May contain flammable or hazardous liquids.",
        "context": "Certain liquids are restricted in transit environments.",
        "action": "Inspection recommended."
    },

    "hammer": {
        "risk": "Blunt object capable of causing physical harm or vandalism.",
        "context": "Unusual in transport areas and may indicate risk.",
        "action": "Monitoring advised."
    }
}

# Convert base64 → image
def decode_image(base64_string):
    img_data = base64.b64decode(base64_string.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# YOLO detection function
def detect_objects(frame):

    results = model(frame, imgsz=640, conf=0.35)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls].lower()
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Determine threat category
        if label in DANGEROUS:
            color = "red"
        elif label in MODERATE:
            color = "yellow"
        else:
            color = "green"

        detections.append({
            "object": label,
            "confidence": round(conf, 2),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "color": color
        })

    return detections

# -----------------------------
# 🔥 RAG Description Generator
# -----------------------------
def generate_rag_description(detections):

    if not detections:
        return "✅ Area appears safe. No suspicious objects detected."

    description_parts = []
    highest_threat = "green"

    for d in detections:

        obj = d["object"]
        color = d["color"]

        if color == "red":
            highest_threat = "red"
        elif color == "yellow" and highest_threat != "red":
            highest_threat = "yellow"

        knowledge = THREAT_KNOWLEDGE.get(obj)

        if knowledge:
            desc = (
                f"Object detected: {obj.upper()}.\n"
                f"Risk: {knowledge['risk']}\n"
                f"Context: {knowledge['context']}\n"
                f"Recommended Action: {knowledge['action']}\n"
            )
        else:
            desc = f"Object detected: {obj}. No known threat intelligence available.\n"

        description_parts.append(desc)

    # Severity headline
    if highest_threat == "red":
        header = "🚨 HIGH THREAT DETECTED\n\n"
    elif highest_threat == "yellow":
        header = "⚠️ MODERATE RISK DETECTED\n\n"
    else:
        header = "✔️ Scene appears safe\n\n"

    return header + "\n".join(description_parts)

# -----------------------------
# API endpoint
# -----------------------------
@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    try:
        data = request.json
        frame_b64 = data.get("frame")

        if not frame_b64:
            return jsonify({"error": "No frame received"}), 400

        frame = decode_image(frame_b64)
        detections = detect_objects(frame)

        rag_description = generate_rag_description(detections)

        # Danger now considers both yellow and red
        danger_flag = any(d["color"] != "green" for d in detections)

        return jsonify({
            "detections": detections,
            "danger": danger_flag,
            "description": rag_description
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, port=3000)
