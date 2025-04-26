import cv2
import numpy as np
from PIL import Image
import os

if __name__ == "__main__":

    # Directory path where the face images are stored.
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n[INFO] Training...")
    # Using a deep learning-based face detector instead of Haar cascade
    protoPath = ".\\real-time-face-recognition\\deploy.prototxt"
    modelPath = ".\\real-time-face-recognition\\res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:
            # Convert image to RGB (as PIL loads images in RGB format)
            PIL_img = Image.open(imagePath).convert('RGB')
            img_numpy = np.array(PIL_img, 'uint8')

            # Convert to OpenCV BGR format
            img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

            # Extract the user ID from the image file name
            id = int(os.path.split(imagePath)[-1].split("-")[1])

            # Prepare input for deep learning face detector
            blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
            detector.setInput(blob)
            detections = detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [img_numpy.shape[1], img_numpy.shape[0], img_numpy.shape[1], img_numpy.shape[0]])
                    (x, y, x2, y2) = box.astype("int")

                    # Convert detected face to grayscale
                    gray_face = cv2.cvtColor(img_bgr[y:y2, x:x2], cv2.COLOR_BGR2GRAY)

                    faceSamples.append(gray_face)
                    ids.append(id)

        return faceSamples, ids


    faces, ids = getImagesAndLabels(path)

    # Train the recognizer with the face samples and corresponding labels
    recognizer.train(faces, np.array(ids))

    # Save the trained model into the current directory
    recognizer.write('trainer.yml')

    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))