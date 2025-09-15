import cv2
import face_recognition
import pickle
import numpy as np
import csv
import os
from datetime import datetime

ENCODINGS_PATH = "face_recognition_service/encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# ✅ Load known encodings
print("[INFO] Loading encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

if len(data["encodings"]) == 0:
    print("[ERROR] No known faces found. Please run train.py first.")
    exit()

# ✅ Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

THRESHOLD = 0.5  # Lower = stricter, 0.45–0.6 works well

# ✅ Keep track of who is already marked today
today = datetime.now().date()
marked_today = set()

# ✅ Make sure attendance file exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])  # header

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, (top, right, bottom, left) in zip(encodings, boxes):
        face_distances = face_recognition.face_distance(data["encodings"], encoding)

        name = "Unknown"
        if len(face_distances) > 0:
            best_idx = np.argmin(face_distances)
            best_distance = face_distances[best_idx]

            if best_distance <= THRESHOLD:
                name = data["names"][best_idx]

                # ✅ Mark attendance only once per student per day
                if (name, today) not in marked_today:
                    now = datetime.now()
                    with open(ATTENDANCE_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, now.date(), now.strftime("%H:%M:%S")])
                    marked_today.add((name, today))
                    print(f"[ATTENDANCE] {name} marked at {now.strftime('%H:%M:%S')}")

        # ✅ Draw green rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
