import cv2
import numpy as np

class YOLOv4Tiny:
    def __init__(self, config_path, weights_path):
        # Load the YOLO model
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        # Prepare the image for YOLO
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Perform a forward pass
        outputs = self.net.forward(self.output_layers)

        # Post-process the outputs
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Filter weak detections
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))
                    detections.append(([x, y, w, h], confidence, class_id))

        return detections
