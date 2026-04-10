import cv2
import os
import numpy as np

data_path = "data/known_faces/giriraj"

faces = []
labels = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

label = 0  # only one known person

for file in os.listdir(data_path):
    img_path = os.path.join(data_path, file)

    if img_path.endswith((".jpg", ".png", ".jpeg")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            faces.append(face_roi)
            labels.append(label)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("models/lbph_model.xml")

print(f"Training completed. Saved model with {len(faces)} face samples.")