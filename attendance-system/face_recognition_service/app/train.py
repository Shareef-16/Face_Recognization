import os
import cv2
import face_recognition
import pickle

# ==== MODIFY THIS PATH TO YOUR DATASET LOCATION ====
# Example: "C:/Users/admin/OneDrive/Desktop/attendance-system/dataset"
DATASET_PATH = os.path.join(os.getcwd(), "dataset")  
ENCODINGS_PATH = os.path.join("face_recognition_service", "encodings.pkl")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] Dataset folder not found: {DATASET_PATH}")
    print("Please create a 'dataset' folder with subfolders for each student containing their images.")
    exit()

print("[INFO] Processing student images...")

known_encodings = []
known_names = []

# Loop through each student folder
for student_name in os.listdir(DATASET_PATH):
    student_folder = os.path.join(DATASET_PATH, student_name)

    if not os.path.isdir(student_folder):
        continue

    for img_file in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_file)

        # Load and convert image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Could not read {img_path}, skipping...")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face encodings
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(student_name)

        print(f"[OK] Encoded {img_file} for {student_name}")

# Save encodings
print("[INFO] Saving encodings...")
data = {"encodings": known_encodings, "names": known_names}

os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"[DONE] Encodings saved to {ENCODINGS_PATH}")
