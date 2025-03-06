import numpy as np
import json
import cv2
import os

def create_directory(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_face_id(directory: str) -> int:
    """Get the next available user ID based on stored images."""
    user_ids = set()
    for filename in os.listdir(directory):
        try:
            number = int(filename.split("-")[1])
            user_ids.add(number)
        except (IndexError, ValueError):
            continue
    return min(set(range(1, max(user_ids, default=0) + 2)) - user_ids)

def save_name(face_id: int, face_name: str, filename: str) -> None:
    """Save user name and ID in a JSON file."""
    names_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            try:
                names_data = json.load(fs)
            except json.JSONDecodeError:
                pass  # Handle empty/corrupt JSON file
    names_data[str(face_id)] = face_name
    with open(filename, 'w') as fs:
        json.dump(names_data, fs, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = 'images'
    cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
    names_json_filename = 'names.json'

    # Create necessary directories
    create_directory(directory)

    # Load Haar cascade file
    cascade_path = os.path.join(os.getcwd(), cascade_classifier_filename)
    faceCascade = cv2.CascadeClassifier('D:\\INSIGHT X\\real-time-face-recognition\\haarcascade_frontalface_default.xml')

    # Open webcam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    # Get user details
    count = 0
    face_name = input('\nEnter user name and press <return> -->  ')
    face_id = get_face_id(directory)
    save_name(face_id, face_name, names_json_filename)

    print('\n[INFO] Initializing face capture. Look at the camera and wait...')

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Faster face detection with optimized parameters
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            count += 1
            filename = os.path.join(directory, f'User-{face_id}-{count}.jpg')
            cv2.imwrite(filename, gray[y:y+h, x:x+w])

            cv2.imshow('image', img)

        # Press 'q' to quit early OR stop after 15 images
        if cv2.waitKey(50) & 0xFF == ord('q') or count >= 30:
            break

    print('\n[INFO] Success! Exiting Program.')

    cam.release()
    cv2.destroyAllWindows()
