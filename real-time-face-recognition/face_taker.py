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


def save_name(face_id: int, face_name: str, roll_no: str, contact: str, filename: str) -> None:
    """Save user details (Name, Roll No, Contact) in a JSON file."""
    names_data = {}

    if os.path.exists(filename):
        with open(filename, 'r') as fs:
            try:
                names_data = json.load(fs)
            except json.JSONDecodeError:
                pass

    # Store details
    names_data[str(face_id)] = {
        "name": face_name,
        "roll_no": roll_no,
        "contact": contact
    }

    with open(filename, 'w') as fs:
        json.dump(names_data, fs, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = 'images'
    cascade_classifier_filename = '.\\real-time-face-recognition\\haarcascade_frontalface_default.xml'
    names_json_filename = 'names.json'

    # Create necessary directories
    create_directory(directory)

    # Load Haar cascade file
    cascade_path = os.path.join(os.getcwd(), cascade_classifier_filename)
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Open webcam
    cam = cv2.VideoCapture(0)
    # cam.set(6, 1280)  # Set width
    # cam.set(8, 720)  # Set height
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get user details
    count = 0
    face_name = input('\nEnter user name: ')
    roll_no = input('Enter roll number: ')
    contact = input('Enter contact number: ')

    face_id = get_face_id(directory)
    save_name(face_id, face_name, roll_no, contact, names_json_filename)

    print('\n[INFO] Initializing face capture. Look at the camera and wait...')
    # Load the deep learning face detector
    net = cv2.dnn.readNetFromCaffe(".\\real-time-face-recognition\\deploy.prototxt", ".\\real-time-face-recognition\\res10_300x300_ssd_iter_140000.caffemodel")

    while True:
        ret, img = cam.read()
        h, w = img.shape[:2]

        # Preprocess image for deep learning model
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Only consider detections with high confidence
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Draw rectangle around detected face
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

                # Save face images
                face_crop = img[y:y1, x:x1]
                count += 1
                cv2.imwrite(f'images/User-{face_id}-{count}.jpg', face_crop)

        cv2.imshow('Face Capture', img)

        if cv2.waitKey(50) & 0xFF == ord('q') or count >= 30:
            break

    print('\n[INFO] Success! Exiting Program.')

    cam.release()
    cv2.destroyAllWindows()
