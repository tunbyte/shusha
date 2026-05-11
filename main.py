import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time


# Load the trained model and label encoder
rf_model = joblib.load("gesture_recognition_model.pkl")  # Replace with your saved model file
label_encoder = joblib.load("label_encoder.pkl")         # Replace with your saved label encoder file

# HandLandmarker Task API setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Çizim için el bağlantıları
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index finger
    (0,9),(9,10),(10,11),(11,12),    # Middle finger
    (0,13),(13,14),(14,15),(15,16),  # Ring finger
    (0,17),(17,18),(18,19),(19,20),  # Pinky
    (5,9),(9,13),(13,17),            # Palm
]

# değişkenler
latest_result = None
committed_text = ""  # oluşan yazı
MODEL_PATH = "hand_landmarker.task"  # mediapipe modeli
current_gesture = None
gesture_start_time = 0
gesture_committed = False  # gesture aktif etme değişkeni
MIN_HOLD_TIME = 0.2  # algılama ve commit süresi


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result



def process_gesture(new_gesture):
    global current_gesture
    global gesture_start_time
    global committed_text

    now = time.time()

    # hiçbir gesture yoksa
    if new_gesture is None:
        current_gesture = None
        gesture_start_time = 0
        return

    # ilk gesture
    if current_gesture is None:
        current_gesture = new_gesture
        gesture_start_time = now
        return

    # gesture değiştiyse
    if new_gesture != current_gesture:
        hold_time = now - gesture_start_time
        # önceki gesture yeterince tutulduysa commit et
        if hold_time >= MIN_HOLD_TIME:
            committed_text += current_gesture
            print("COMMIT:", current_gesture)

        # yeni gesture'a geç
        current_gesture = new_gesture
        gesture_start_time = now



# HandLandmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

decoded_label = ""  # çözülen harfin tutulduğu değişken
# webcam akışı başlatılıyor
cap = cv2.VideoCapture(0)
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )


        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        if latest_result and latest_result.hand_landmarks:
            h, w, _ = frame.shape

            for i, hand_landmarks in enumerate(latest_result.hand_landmarks):
                if latest_result.hand_landmarks:
                  handedness = latest_result.handedness[i][0].category_name

                for start_idx, end_idx in HAND_CONNECTIONS:
                    sx = int(hand_landmarks[start_idx].x * w)
                    sy = int(hand_landmarks[start_idx].y * h)
                    ex = int(hand_landmarks[end_idx].x * w)
                    ey = int(hand_landmarks[end_idx].y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)

                # Draw keypoints
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # ---- Hareket Tanıma ----
                # Tahmin için el referans noktalarını normalleştirin
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
                if handedness == "Left":
                    landmarks[:, 0] = 1.0 - landmarks[:, 0]
                wrist = landmarks[0]  # Bilek koordinatları (nokta 0)
                landmarks -= wrist    # Bileğe doğru merkez
                scale = np.linalg.norm(landmarks[9])  # Parmak köküne olan mesafe (9 numaralı referans noktası)
                if scale == 0:
                    continue  # Sıfıra bölme işleminden kaçının
                landmarks /= scale
                features = landmarks.flatten()  # Özellik vektörüne (1 boyutlu dizi) dönüştürün

                # Modeli kullanarak hareketi tahmin edin
                feature_names = [f"{c}{i}" for i in range(21) for c in ["x", "y", "z"]]
                features_df = pd.DataFrame([features], columns=feature_names)
                prediction = rf_model.predict(features_df)
                decoded_label = label_encoder.inverse_transform(prediction)[0]

                # jest durum makinesi fonksiyonu
                process_gesture(decoded_label)

                # Tahmini çerçeve üzerinde görüntüleyin
                cv2.putText(frame, f"Current: {current_gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)

                cv2.putText(frame,
                            f"Text: {committed_text}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA)

        else:
            process_gesture(None)  # el görünmüyorsa boş geçilir
        cv2.imshow('Hand Gesture Recognition', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

