import time
import cv2
from object_detector import ObjectDetector
from alert import send_telegram_alert

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.xml")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# YOLO object detector
object_detector = ObjectDetector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening camera")
    exit()

print("Press q to quit")

# Alert cooldown
last_alert_time = 0
alert_cooldown = 10  # seconds

# Identity locking variables
prediction_history = []
locked_status = None
no_face_counter = 0
lock_frames_required = 15
unlock_frames_required = 5

# YOLO throttling
frame_count = 0
cached_threats = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    person_status = "No Face"

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        no_face_counter += 1
        if no_face_counter >= unlock_frames_required:
            locked_status = None
            prediction_history.clear()
    else:
        no_face_counter = 0

    # Face recognition
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        try:
            _, confidence = recognizer.predict(face_roi)

            if confidence < 75:
                current_prediction = "Known"
            else:
                current_prediction = "Unknown"

            if locked_status is None:
                prediction_history.append(current_prediction)

                if len(prediction_history) > lock_frames_required:
                    prediction_history.pop(0)

                if len(prediction_history) == lock_frames_required:
                    known_count = prediction_history.count("Known")
                    unknown_count = prediction_history.count("Unknown")

                    if known_count >= 13:
                        locked_status = "Known"
                    elif unknown_count >= 13:
                        locked_status = "Unknown"

            person_status = locked_status if locked_status is not None else current_prediction
            color = (0, 255, 0) if person_status == "Known" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{person_status} ({confidence:.1f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        except Exception:
            pass

    # Run YOLO only every 5th frame
    if frame_count % 5 == 0:
        cached_threats = object_detector.detect(frame)

    threats = cached_threats

    if threats:
        cv2.putText(
            frame,
            f"Threat: {', '.join(threats)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # On-screen alert status
    alert_text = "SAFE"

    if person_status == "Unknown" and threats:
        alert_text = "HIGH THREAT"
    elif person_status == "Unknown":
        alert_text = "WARNING"
    elif person_status == "Known" and threats:
        alert_text = "THREAT ALERT"

    cv2.putText(
        frame,
        f"Status: {alert_text}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if alert_text != "SAFE" else (0, 255, 0),
        2
    )

    # Telegram alert with cooldown
    current_time = time.time()

    if current_time - last_alert_time > alert_cooldown:
        if person_status == "Unknown" and threats:
            send_telegram_alert(
                f"🔥 HIGH THREAT: Unknown person with {', '.join(threats)} detected!"
            )
            last_alert_time = current_time

        elif person_status == "Unknown" and not threats:
            send_telegram_alert("⚠️ WARNING: Unknown person detected!")
            last_alert_time = current_time

        elif person_status == "Known" and threats:
            send_telegram_alert(
                f"🚨 THREAT ALERT: Suspicious object detected - {', '.join(threats)}"
            )
            last_alert_time = current_time

    cv2.imshow("Smart Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()