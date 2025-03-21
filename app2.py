from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import json
import base64

app = Flask(__name__)

# Directory for saving images
IMAGE_DIR = 'images'
os.makedirs(IMAGE_DIR, exist_ok=True)

# Save user names in a JSON file
def save_name(face_id: int, face_name: str, filename='names.json'):
    names_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            try:
                names_data = json.load(fs)
            except json.JSONDecodeError:
                pass
    names_data[str(face_id)] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_data, fs, ensure_ascii=False, indent=4)

# Get next available user ID
def get_face_id(directory: str) -> int:
    user_ids = set()
    for filename in os.listdir(directory):
        try:
            number = int(filename.split("-")[1])
            user_ids.add(number)
        except (IndexError, ValueError):
            continue
    return min(set(range(1, max(user_ids, default=0) + 2)) - user_ids)

@app.route('/')
def index():
    return render_template('header.html')

@app.route('/capture', methods=['POST'])
def capture_image():
    try:
        # Get image and user name from request
        data = request.get_json()
        image_data = data['image']
        face_name = data['name']

        # Decode base64 image
        image_data = image_data.split(',')[1]
        img = base64.b64decode(image_data)
        np_img = np.frombuffer(img, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Get the next face ID and save the image
        face_id = get_face_id(IMAGE_DIR)
        image_filename = f"{IMAGE_DIR}/User-{face_id}.jpg"
        cv2.imwrite(image_filename, image)

        # Save the name and ID in JSON
        save_name(face_id, face_name)

        return jsonify({'success': True, 'message': 'Image saved successfully!'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
