from ultralytics import YOLO

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.threat_objects = ["knife", "scissors", "baseball bat"]
        self.conf_threshold = 0.65

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detected_threats = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                confidence = float(box.conf[0])

                if label in self.threat_objects and confidence >= self.conf_threshold:
                    detected_threats.append(label)

        return list(set(detected_threats))