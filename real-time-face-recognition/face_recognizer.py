import cv2
import numpy as np
import json
import os

if __name__ == "__main__":
    # Create LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Load the trained model
    recognizer.read('trainer.yml')

    # Load Deep Learning Face Detector
    net = cv2.dnn.readNetFromCaffe(".\\real-time-face-recognition\\deploy.prototxt", ".\\real-time-face-recognition\\res10_300x300_ssd_iter_140000.caffemodel")

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

    # Start video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)  # Set width
    cam.set(4, 720)  # Set height

    while True:
        ret, img = cam.read()
        h, w = img.shape[:2]

        # Convert to blob for DNN model
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # Higher threshold to reduce false positives
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Draw rectangle around detected face
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

                # Convert to grayscale for recognition
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y:y1, x:x1]

                # Recognize face
                try:
                    id, confidence = recognizer.predict(face_roi)
                    # print(names)
                    # print(recognizer.predict(face_roi))
                    if confidence < 50:  # Lower confidence = better match
                        name = names[id-1]
                        confidence_text = f"  {round(100 - confidence)}%"
                    else:
                        name = "Unknown"
                        confidence_text = "N/A"
                except Exception as e:
                    print(e)
                    name = "Unknown"
                    confidence_text = "N/A"

                # Display name & confidence
                cv2.putText(img, name, (x, y - 10), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x, y1 + 20), font, 1, (255, 255, 0), 1)

        # Show camera output
        cv2.imshow('Face Recognition', img)

        # Press ESC to exit
        if cv2.waitKey(10) & 0xFF == 27:
            break

    print("\n[INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()