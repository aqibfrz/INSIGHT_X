# from flask import Flask, render_template, Response
# from camra import Video
# app=Flask(__name__)


# @app.route('/')
# def index():
#     return render_template('index.html')

# def gen(camra):
#     while True:
#         frame=camra.get_frame()
#         yield(b'--frame\r\n'b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
# @app.route('/video')
# def video():
#     return Response(gen(Video()),
#     mimetype='multipart/x-mixed-replace; boundary=frame')

# app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import json
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Load LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load DNN Face Detector
net = cv2.dnn.readNetFromCaffe("./real-time-face-recognition/deploy.prototxt", "./real-time-face-recognition/res10_300x300_ssd_iter_140000.caffemodel")

# Font for displaying text on the image
font = cv2.FONT_HERSHEY_SIMPLEX

# Load user names from JSON
names = ['Unknown']
try:
    with open('names.json', 'r') as fs:
        names_data = json.load(fs)
        names = list(names_data.values())
except Exception as e:
    print(f"[ERROR] Unable to load names.json: {e}")

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Process the image received from the front-end."""
    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the base64 image string to an image
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(img_data))
    img = np.array(img)

    # Perform face detection
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Recognize faces
    result = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Higher threshold to reduce false positives
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Convert to grayscale for recognition
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y1, x:x1]

            try:
                id, confidence = recognizer.predict(face_roi)
                if confidence < 50:  # Lower confidence = better match
                    name = names[id-1]
                else:
                    name = "Unknown"
                result.append({"name": name, "confidence": confidence})
            except Exception as e:
                result.append({"name": "Unknown", "confidence": "N/A"})

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
