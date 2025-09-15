import os
import cv2
import numpy as np
from PIL import Image


class FaceTrainer:
    def __init__(self, images_path='./images/', proto_path=None,
                 model_path=None, model_save_path=None):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(repo_root, 'real-time-face-recognition')
        self.images_path = os.path.join(repo_root, images_path)
        self.proto_path = proto_path or os.path.join(models_dir, 'deploy.prototxt')
        self.model_path = model_path or os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        # Save unified trainer to repo root for all components
        self.model_save_path = model_save_path or os.path.join(repo_root, 'trainer.yml')

        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Load face detector model
        self.detector = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def get_images_and_labels(self):
        if not os.path.exists(self.images_path):
            return [], []
        image_paths = [os.path.join(self.images_path, f) for f in os.listdir(self.images_path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('RGB')
            img_numpy = np.array(PIL_img, 'uint8')
            img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

            # Extract the ID from the filename assuming format something like: user-<id>-something.jpg
            try:
                id = int(os.path.split(image_path)[-1].split("-")[1])
            except (IndexError, ValueError) as e:
                print(f"[WARNING] Skipping file with unexpected format: {image_path}")
                continue

            blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
            self.detector.setInput(blob)
            detections = self.detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [img_numpy.shape[1], img_numpy.shape[0], img_numpy.shape[1], img_numpy.shape[0]])
                    (x, y, x2, y2) = box.astype("int")
                    gray_face = cv2.cvtColor(img_bgr[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
                    face_samples.append(gray_face)
                    ids.append(id)

        return face_samples, ids

    def train(self):
        print("\n[INFO] Training face recognizer...")
        faces, ids = self.get_images_and_labels()
        if not faces or not ids:
            print("[ERROR] No faces or IDs found for training.")
            return

        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write(self.model_save_path)
        print(
            f"\n[INFO] Training complete. {len(np.unique(ids))} faces trained and model saved to {self.model_save_path}")


if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train()
