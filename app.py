from flask import Flask, render_template, jsonify, request
import cv2
import time
import threading
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
import json
from plyer import notification

app = Flask(__name__)

# === Constants ===
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.23
MONITOR_DURATION = 30
MIN_BRIGHTNESS = 30
SETTINGS_FILE = "settings.json"

# === Global Variables ===
blink_count = 0
blink_in_progress = False
paused = False
face_present = False
monitor_running = True
initial_brightness = sbc.get_brightness(display=0)[0]
current_brightness = initial_brightness
start_time = time.time()

# === MediaPipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === Settings Save/Load ===
def save_settings():
    with open(SETTINGS_FILE, "w") as f:
        json.dump({"brightness": current_brightness}, f)

def load_settings():
    global initial_brightness, current_brightness
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            initial_brightness = data.get("brightness", initial_brightness)
            current_brightness = initial_brightness
    except:
        pass

# === Alert ===
def speak_alert(text):
    try:
        notification.notify(
            title="Eye Alert",
            message=text,
            timeout=5
        )
    except Exception as e:
        print("Alert failed:", e)

# === EAR Calculation ===
def calculate_ear(landmarks, indices):
    p = [landmarks[i] for i in indices]
    vertical1 = np.linalg.norm([p[1].x - p[5].x, p[1].y - p[5].y])
    vertical2 = np.linalg.norm([p[2].x - p[4].x, p[2].y - p[4].y])
    horizontal = np.linalg.norm([p[0].x - p[3].x, p[0].y - p[3].y])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# === Brightness Logic ===
def adjust_brightness(level):
    global current_brightness
    try:
        sbc.set_brightness(level)
        current_brightness = level
        save_settings()
    except Exception as e:
        print("Brightness error:", e)

def handle_brightness():
    global blink_count, initial_brightness, current_brightness
    if current_brightness <= MIN_BRIGHTNESS:
        return

    if blink_count > 15:
        adjust_brightness(initial_brightness)
    elif 10 <= blink_count <= 15:
        adjust_brightness(max(initial_brightness - 10, MIN_BRIGHTNESS))
    elif 1 <= blink_count < 10:
        adjust_brightness(max(initial_brightness - 20, MIN_BRIGHTNESS))
    elif blink_count == 0 and face_present:
        adjust_brightness(max(initial_brightness - 30, MIN_BRIGHTNESS))
        speak_alert("உங்கள் கண்கள் சிமிட்டவில்லை. தயவுசெய்து கண் சிமிட்டுங்கள்.")
    blink_count = 0

# === Monitoring Thread ===
def monitor():
    global blink_count, blink_in_progress, start_time, paused, face_present, initial_brightness, current_brightness, monitor_running
    cap = cv2.VideoCapture(0)
    
    while monitor_running:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        if not paused:
            results = face_mesh.process(rgb)
            face_present = results.multi_face_landmarks is not None
            if face_present:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                avg_ear = (left_ear + right_ear) / 2

                if avg_ear < BLINK_THRESHOLD:
                    if not blink_in_progress:
                        blink_count += 1
                        blink_in_progress = True
                else:
                    blink_in_progress = False

            # Detect manual brightness change
            new_brightness = sbc.get_brightness(display=0)[0]
            if new_brightness != current_brightness:
                initial_brightness = new_brightness
                current_brightness = new_brightness

        if time.time() - start_time >= MONITOR_DURATION:
            handle_brightness()
            start_time = time.time()

    cap.release()

# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    elapsed = int(time.time() - start_time)
    return jsonify({
        'blink_count': blink_count,
        'brightness': current_brightness,
        'timer': elapsed,
        'paused': paused,
        'face_present': face_present
    })

@app.route('/control', methods=['POST'])
def control():
    global paused, blink_count, monitor_running, current_brightness
    
    action = request.json.get('action')
    
    if action == 'toggle_pause':
        paused = not paused
    elif action == 'reset':
        blink_count = 0
    elif action == 'set_brightness':
        brightness = request.json.get('brightness')
        if brightness is not None:
            adjust_brightness(brightness)
    elif action == 'shutdown':
        monitor_running = False
        save_settings()
    
    return jsonify({'success': True})

# === Main ===
if __name__ == "__main__":
    load_settings()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    app.run(debug=False, port=5000)
