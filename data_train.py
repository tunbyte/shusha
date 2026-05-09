import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

DATASET_DIR = "asl_alphabet_train"
OUTPUT_CSV  = "dataset.csv"
SKIP_LABELS = {"space", "del", "nothing"}

# CSV başlık satırı: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

def normalize_landmarks(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])

    # Move all lines according to the wrist
    coords = []
    for lm in landmarks:
        coords.append([lm.x - wrist[0], lm.y - wrist[1], lm.z - wrist[2]])
    coords = np.array(coords)

    # Scale: nokta 9 (orta parmak dibi) ile bilek arası mesafe
    scale = np.linalg.norm(coords[9])
    if scale == 0:
        return None  # geçersiz el, atla

    coords /= scale
    return coords.flatten().tolist()  # 63 değer


skipped = 0
saved   = 0

with HandLandmarker.create_from_options(options) as landmarker:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        labels = sorted(os.listdir(DATASET_DIR))

        for label in labels:
            if label.lower() in SKIP_LABELS:
                print(f"  [{label}] atlandı")
                continue

            label_dir = os.path.join(DATASET_DIR, label)
            if not os.path.isdir(label_dir):
                continue

            images = list(Path(label_dir).glob("*.jpg")) + \
                     list(Path(label_dir).glob("*.png")) + \
                     list(Path(label_dir).glob("*.JPG"))

            label_saved   = 0
            label_skipped = 0

            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    label_skipped += 1
                    continue

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                )

                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    label_skipped += 1
                    continue

                features = normalize_landmarks(result.hand_landmarks[0])
                if features is None:
                    label_skipped += 1
                    continue

                writer.writerow(features + [label])
                label_saved += 1

            saved   += label_saved
            skipped += label_skipped
            print(f"  [{label}] ✓ {label_saved} kaydedildi, {label_skipped} atlandı")

print(f"\nTamamlandı! Toplam: {saved} örnek kaydedildi, {skipped} atlandı.")
print(f"Çıktı: {OUTPUT_CSV}")