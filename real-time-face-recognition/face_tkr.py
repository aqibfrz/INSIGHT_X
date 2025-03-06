import pickle
import cv2
import json
import os
import numpy as np
from PIL import Image

def get_images_and_labels(directory: str):
    """
    Load face images and corresponding labels from the given directory.

    Returns:
        faces (list): List of face samples as numpy arrays.
        ids (list): List of corresponding labels (face IDs).
    """
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        try:
            # Convert image to grayscale
            PIL_img = Image.open('D:\\INSIGHT X\\images').convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract user ID from filename (Users-ID-Count.jpg)
            id = int(os.path.split('D:\\INSIGHT X\\images')[-1].split("-")[1])

            face_samples.append(img_numpy)
            ids.append(id)
        except Exception as e:
            print(f"Skipping {'images'}: {e}")

    return face_samples,ids

# Get images and labels
faces, ids = get_images_and_labels('D:\\INSIGHT X\\images')

# Load name-ID mapping
with open('names.json', "r") as f:
    names_dict = json.load(f)

# Save dataset as a pickle file
with open("taker.pkl", "wb") as f:
    pickle.dump({"faces": faces, "ids": ids, "names": names_dict}, f)

print("\n[INFO] Face dataset saved as dataset.pkl")

