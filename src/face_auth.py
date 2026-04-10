import os
import cv2
import numpy as np
from deepface import DeepFace


class FaceAuthenticator:
    def __init__(self, known_faces_dir):
        self.known_faces_dir = known_faces_dir
        self.reference_image = self._get_reference_image()
        self.reference_embedding = self._build_reference_embedding()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _get_reference_image(self):
        for file in os.listdir(self.known_faces_dir):
            path = os.path.join(self.known_faces_dir, file)
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                return path
        return None

    def _build_reference_embedding(self):
        if self.reference_image is None:
            return None

        try:
            result = DeepFace.represent(
                img_path=self.reference_image,
                model_name="OpenFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception:
            return None

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def recognize(self, face_crop):
        if self.reference_embedding is None:
            return "Unknown"

        try:
            result = DeepFace.represent(
                img_path=face_crop,
                model_name="OpenFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            live_embedding = np.array(result[0]["embedding"], dtype=np.float32)

            distance = np.linalg.norm(self.reference_embedding - live_embedding)

            if distance < 0.7:
                return "Known"
            return "Unknown"

        except Exception:
            return "Unknown"