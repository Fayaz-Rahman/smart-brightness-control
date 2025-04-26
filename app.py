# app.py

import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Constants
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.23

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_ear(landmarks, indices):
    p = [landmarks[i] for i in indices]
    vertical1 = np.linalg.norm([p[1].x - p[5].x, p[1].y - p[5].y])
    vertical2 = np.linalg.norm([p[2].x - p[4].x, p[2].y - p[4].y])
    horizontal = np.linalg.norm([p[0].x - p[3].x, p[0].y - p[3].y])
    return (vertical1 + vertical2) / (2.0 * horizontal)

@app.route('/', methods=['GET', 'POST'])
def index():
    blink_detected = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = 'EyeLogo.ico'
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Process the image
            image = cv2.imread(filepath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                avg_ear = (left_ear + right_ear) / 2

                if avg_ear < BLINK_THRESHOLD:
                    blink_detected = "Blink Detected"
                else:
                    blink_detected = "Eyes Open"

    return render_template('index.html', blink=blink_detected)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
