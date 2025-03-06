import cv2
import numpy as np
import json
import os

if __name__ == "__main__":
    # Create LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load the trained model
    recognizer.read('trainer.yml')

    # Get Haar cascade file path dynamically
    cascade_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier('D:\\INSIGHT X\\real-time-face-recognition\\haarcascade_frontalface_default.xml')

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
    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    # Minimum face size
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recognize face
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Lower confidence = better match
            if confidence < 51:
                try:
                    name = names[id]
                    confidence_text = "  {0}%".format(round(100 - confidence))
                except IndexError:
                    name = "Unknown"
                    confidence_text = "N/A"
            else:
                name = "Unknown"
                confidence_text = "N/A"

            # Display name & confidence
            cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # Show camera output
        cv2.imshow('Face Recognition', img)

        # Press ESC to exit
        if cv2.waitKey(10) & 0xFF == 27:
            break

    print("\n[INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()
    
