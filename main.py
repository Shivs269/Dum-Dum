import cv2
import logging
import numpy as np
import datetime
import base64
import json
import os
from huggingface_hub import InferenceClient

# Initialize Hugging Face API client
client = InferenceClient(api_key="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")

# Logging setup
logging.basicConfig(level=logging.INFO)

# YOLOv4-Tiny Configuration
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CONFIG = "yolov4-tiny.cfg"
YOLO_CLASSES = "coco.names"

# Create 'frames' folder to save images
FRAMES_DIR = "frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# Load YOLOv4-Tiny Model
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()


# Motion detection function
def detect_motion(frame, prev_frame, threshold=50):
    if prev_frame is None:
        return False, None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, gray_prev_frame)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_detected = cv2.countNonZero(motion_mask) > threshold
    return motion_detected, motion_mask


# Frame sharpness evaluation
def evaluate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


# Detect objects, draw bounding boxes, and return detections
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    # Parse YOLO output
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))  # Convert np.int64 to native int
            confidence = float(scores[class_id])  # Convert np.float32 to native float
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []

    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"
            color = (0, 255, 0)
            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append({
                "label": class_names[class_ids[i]],
                "confidence_score": confidences[i],
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "priority": "urgent" if class_names[class_ids[i]] in ["person", "car"] else "medium"
            })
    return frame, detections


# Process frame with Qwen API
def process_frame_with_qwen(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-2B-Instruct",
            messages=messages,
            max_tokens=500
        )
        if response and response.choices:
            return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling Qwen API: {e}")
    return None


# Save frame to disk and return file name
def save_frame(frame, frame_id):
    frame_name = f"frame_{frame_id}.jpg"
    frame_path = os.path.join(FRAMES_DIR, frame_name)
    cv2.imwrite(frame_path, frame)
    return frame_name


# Main pipeline
def main_pipeline():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        logging.error("Error: Cannot access the video source.")
        return

    prev_frame = None
    best_frame = None
    best_sharpness = 0
    frame_count = 0
    all_frame_data = []

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            logging.error("Error: Invalid frame received from video source.")
            break

        # Detect motion
        motion_detected, _ = detect_motion(frame, prev_frame)
        prev_frame = frame.copy()

        # Track sharpest frame if motion is detected
        if motion_detected:
            sharpness = evaluate_sharpness(frame)
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame.copy()

        # Detect and annotate objects
        annotated_frame, detections = detect_objects(frame)

        # Save the best frame every 100 frames
        if frame_count % 100 == 0 and best_frame is not None:
            frame_file = save_frame(best_frame, frame_count)
            description = process_frame_with_qwen(best_frame)
            frame_data = {
                "frame_id": frame_count,
                "timestamp": str(datetime.datetime.now()),
                "frame_file": frame_file,
                "description": description,
                "detections": detections
            }
            all_frame_data.append(frame_data)
            logging.info(f"Processed frame {frame_count}: {json.dumps(frame_data, indent=4)}")
            best_frame = None
            best_sharpness = 0

        # Display video stream
        cv2.imshow("Video Stream with Bounding Boxes", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting the video stream.")
            break

        frame_count += 1

    # Save all collected data to a txt file
    with open("processed_data.txt", "w") as file:
        json.dump(all_frame_data, file, indent=4)

    logging.info("All frame data saved to processed_data.txt.")
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.info("Starting the real-time video stream pipeline...")
    main_pipeline()
