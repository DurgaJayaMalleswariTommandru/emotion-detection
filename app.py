from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("model.h5")

# Emotion labels (must match training order)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    data = request.json["image"]

    # Remove base64 header
    image_data = data.split(",")[1]

    # Decode base64 to bytes
    img_bytes = base64.b64decode(image_data)

    # Convert bytes to numpy array
    np_arr = np.frombuffer(img_bytes, np.uint8)

    # Decode image using OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        return jsonify({"emotion": emotion})

    return jsonify({"emotion": "No Face Detected"})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
