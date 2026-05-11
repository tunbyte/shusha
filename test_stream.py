import cv2
import time
import mediapipe as mp



#  Declaring variables
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # başparmak
    (0,5),(5,6),(6,7),(7,8),         # işaret parmağı
    (0,9),(9,10),(10,11),(11,12),    # orta parmak
    (0,13),(13,14),(14,15),(15,16),  # yüzük parmağı
    (0,17),(17,18),(18,19),(19,20),  # serçe
    (5,9),(9,13),(13,17),            # avuç içi
]

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


latest_result = None
MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    result_callback=result_callback,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
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

            for hand in latest_result.hand_landmarks:
                # Bağlantıları çiz
                for start_idx, end_idx in HAND_CONNECTIONS:
                    sx = int(hand[start_idx].x * w)
                    sy = int(hand[start_idx].y * h)
                    ex = int(hand[end_idx].x * w)
                    ey = int(hand[end_idx].y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)

                # Keypoint'leri çiz
                for i, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (150, 255, 0), -1)


        frame = cv2.flip(frame, 1)

        cv2.imshow("MediaPipe Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
