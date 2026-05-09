import cv2
import mediapipe as mp
import numpy as np
import joblib

rf_model = joblib.load("gesture_recognition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

IMAGE_PATH = "asl_alphabet_test/asl_alphabet_test/B_test.jpg"

frame = cv2.imread(IMAGE_PATH)
h, w, _ = frame.shape

with HandLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name

            for start_idx, end_idx in HAND_CONNECTIONS:
                sx = int(hand_landmarks[start_idx].x * w)
                sy = int(hand_landmarks[start_idx].y * h)
                ex = int(hand_landmarks[end_idx].x * w)
                ey = int(hand_landmarks[end_idx].y * h)
                cv2.line(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
            if handedness == "Left":
                landmarks[:, 0] = 1.0 - landmarks[:, 0]

            wrist = landmarks[0]
            landmarks -= wrist
            scale = np.linalg.norm(landmarks[9])
            if scale > 0:
                landmarks /= scale

            features = landmarks.flatten()
            prediction = rf_model.predict([features])
            decoded_label = label_encoder.inverse_transform(prediction)[0]

            cv2.putText(frame, f"Gesture: {decoded_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    else:
        print("El tespit edilemedi.")

    cv2.imshow("Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()