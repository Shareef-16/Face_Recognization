import os
import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime

# Paths
DATASET_PATH = os.path.join(os.getcwd(), "dataset")
ENCODINGS_PATH = os.path.join("face_recognition_service", "encodings.pkl")
ATTENDANCE_FILE = os.path.join(os.getcwd(), "attendance.csv")

# Ensure dataset folder exists
os.makedirs(DATASET_PATH, exist_ok=True)

# Attendance CSV setup
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])  # header

# Load existing encodings if available
def load_encodings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(data):
    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

# ====================== Register New Student ======================
def register_student():
    student_name = input("Enter new student name/ID: ").strip()
    student_folder = os.path.join(DATASET_PATH, student_name)
    os.makedirs(student_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    print(f"[INFO] Capturing image for {student_name}. Press 's' to take snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Register Student - Press 's' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            img_path = os.path.join(student_folder, "img1.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Image saved to {img_path}")
            break
        elif key == ord("q"):
            print("[INFO] Registration aborted")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # Update encodings immediately
    print("[INFO] Updating encodings...")
    train_student(student_name)
    print(f"[DONE] Student {student_name} registered successfully!")

# ====================== Train Student ======================
def train_student(student_name=None):
    data = load_encodings()
    if student_name:
        students = [student_name]
    else:
        students = os.listdir(DATASET_PATH)

    for student in students:
        student_folder = os.path.join(DATASET_PATH, student)
        if not os.path.isdir(student_folder):
            continue

        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                data["encodings"].append(encoding)
                data["names"].append(student)
            print(f"[OK] Encoded {img_file} for {student}")

    save_encodings(data)

# ====================== Mark Attendance ======================
def mark_attendance():
    data = load_encodings()
    if len(data["encodings"]) == 0:
        print("[ERROR] No known faces. Register students first.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    THRESHOLD = 0.5
    today = datetime.now().date()
    marked_today = set()

    print("[INFO] Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

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
                    if (name, today) not in marked_today:
                        now = datetime.now()
                        with open(ATTENDANCE_FILE, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([name, now.date(), now.strftime("%H:%M:%S")])
                        marked_today.add((name, today))
                        print(f"[ATTENDANCE] {name} marked at {now.strftime('%H:%M:%S')}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====================== Main Menu ======================
def main():
    while True:
        print("\n===== Attendance System =====")
        print("1. Register new student")
        print("2. Mark attendance")
        print("3. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            register_student()
        elif choice == "2":
            mark_attendance()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
