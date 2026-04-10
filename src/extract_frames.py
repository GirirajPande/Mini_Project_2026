import cv2
import os

video_path = "data/my_video.mp4"
output_folder = "data/known_faces/giriraj"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0
frame_skip = 10  # save every 10th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        file_name = os.path.join(output_folder, f"face_{saved_count}.jpg")
        cv2.imwrite(file_name, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Done. Saved {saved_count} frames in {output_folder}")