import cv2
import numpy as np
from PIL import Image
import os
import pickle

# Directory path where the face images are stored
path = './images/'

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade face detector
cascade_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier('D:\\INSIGHT X\\real-time-face-recognition\\haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    """
    Load face images and corresponding labels from the given directory path.

    Returns:
        list: List of face samples.
        list: List of corresponding labels.
    """
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            # Ensure file is an image
            if not imagePath.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            # Convert image to grayscale
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract the user ID from the image file name
            id = int(os.path.split(imagePath)[-1].split("-")[1])

            # Detect faces in the grayscale image
            faces = detector.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        except Exception as e:
            print(f"Skipping file {imagePath}: {e}")

    return faceSamples, ids

if __name__ == "__main__":
    print("\n[INFO] Training...")

    # Get face samples and corresponding labels
    faces, ids = getImagesAndLabels(path)

    if len(faces) == 0:
        print("\n[ERROR] No faces detected. Check the dataset.")
    else:
        # Train the recognizer with the face samples and corresponding labels
        recognizer.train(faces, np.array(ids))
        
        # Save the trained model as a YAML file
        recognizer.write('trainer.yml')
        
        # Save the trained model as a pickle file
        model_data = {
            "radius": recognizer.getRadius(),
            "neighbors": recognizer.getNeighbors(),
            "grid_x": recognizer.getGridX(),
            "grid_y": recognizer.getGridY(),
            "histograms": recognizer.getHistograms(),
            "labels": recognizer.getLabels()
        }

        with open("trainer.pkl", "wb") as f:
            pickle.dump(model_data, f)

        print("\n[INFO] {0} faces trained. Model saved as trainer.pkl".format(len(np.unique(ids))))
