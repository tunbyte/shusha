# SHUSHA 1.0

import base64
import time
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import joblib

# web service
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shusha'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Model yükleme
rf_model = joblib.load("gesture_recognition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ── MediaPipe HandLandmarker ───────────────────────────────────
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

MODEL_PATH   = "hand_landmarker.task"
MIN_HOLD_TIME = 0.2

# Oturum durumu (her WebSocket bağlantısı için ayrı)
# değişkenler
state = {
    "latest_result": None,
    "committed_text":  "",
    "current_gesture": None,
    "gesture_start": 0,
    "landmarker": None,
}

def _result_callback(result, output_image, timestamp_ms):
    state["latest_result"] = result

def _init_landmarker():
    if state["landmarker"] is not None:
        return
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=_result_callback,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    state["landmarker"] = HandLandmarker.create_from_options(options)
    print("HandLandmarker başlatıldı.")

def _process_gesture(new_gesture, sid):
    now = time.time()
    if new_gesture is None:
        state["current_gesture"] = None
        state["gesture_start"]   = 0
        return

    if state["current_gesture"] is None:
        state["current_gesture"] = new_gesture
        state["gesture_start"]   = now
        return

    if new_gesture != state["current_gesture"]:
        hold_time = now - state["gesture_start"]
        if hold_time >= MIN_HOLD_TIME:
            # Özel komutlar
            if state["current_gesture"] == "space":
                state["committed_text"] += " "
            elif state["current_gesture"] == "del":
                state["committed_text"] = state["committed_text"][:-1]
            else:
                state["committed_text"] += state["current_gesture"]
            print(f"[COMMIT] '{state['current_gesture']}' → {state['committed_text']}")
            # Anlık güncellemeyi tarayıcıya gönder
            socketio.emit("text_update", {
                "text":    state["committed_text"],
                "gesture": state["current_gesture"],
            }, to=sid)

        state["current_gesture"] = new_gesture
        state["gesture_start"]   = now

# arayüz
@app.route("/")
def index():
    return render_template("index.html")

# ── WebSocket bağlantıları ─────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    _init_landmarker()
    print(f"Bağlantı kuruldu:")
    emit("connected", {"msg": "Sunucuya bağlandı."})

@socketio.on("disconnect")
def on_disconnect():
    print(f"Bağlantı kesildi")

@socketio.on("frame")
def on_frame(data):
    """
    Tarayıcıdan gelen base64 JPEG kare.
    data = { "image": "data:image/jpeg;base64,..." }
    """
    sid = request_sid()
    try:
        # base64 → numpy array
        b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        h, w, _ = frame.shape

        # MediaPipe görüntüsüne çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        state["landmarker"].detect_async(mp_image, timestamp)

        result = state["latest_result"]
        overlay_data = []  # çizim noktaları / bağlantılar için

        if result and result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[i][0].category_name

                # Normalleştirme
                lms = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
                if handedness == "Left":
                    lms[:, 0] = 1.0 - lms[:, 0]
                wrist = lms[0]
                lms  -= wrist
                scale = np.linalg.norm(lms[9])
                if scale == 0:
                    continue
                lms /= scale

                # Tahmin
                feature_names = [f"{c}{i}" for i in range(21) for c in ["x","y","z"]]
                features_df   = pd.DataFrame([lms.flatten()], columns=feature_names)
                pred          = rf_model.predict(features_df)
                gesture       = label_encoder.inverse_transform(pred)[0]

                _process_gesture(gesture, sid)

                # Landmark koordinatları (normalize 0-1) → tarayıcıya gönder
                points = [{"x": lm.x, "y": lm.y} for lm in hand_landmarks]
                overlay_data.append({
                    "points":      points,
                    "connections": HAND_CONNECTIONS,
                    "gesture":     gesture,
                    "handedness":  handedness,
                })

            emit("overlay", {
                "hands":          overlay_data,
                "current_gesture": state["current_gesture"],
                "committed_text":  state["committed_text"],
            })
        else:
            _process_gesture(None, sid)
            emit("overlay", {
                "hands":          [],
                "current_gesture": None,
                "committed_text":  state["committed_text"],
            })

    except Exception as e:
        print(f"[HATA] on_frame: {e}")

@socketio.on("clear_text")
def on_clear_text():
    state["committed_text"]  = ""
    state["current_gesture"] = None
    state["gesture_start"]   = 0
    emit("text_cleared")

def request_sid():
    return request.sid

# ── Uygulama başlatma ──────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)

