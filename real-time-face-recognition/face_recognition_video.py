import cv2
import numpy as np
import json
import os


def recognize_faces_in_video(video_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    trainer_yml_path = os.path.join(repo_root, 'trainer.yml')
    if not os.path.exists(trainer_yml_path):
        print(f"[ERROR] Trainer file 'trainer.yml' not found at: {trainer_yml_path}")
        return

    # Ensure names.json exists
    names_path = os.path.join(repo_root, 'names.json')
    if not os.path.exists(names_path):
        print("[ERROR] Names file 'names.json' not found.")
        return

    # Face detection model paths
    proto_path = os.path.join(os.path.dirname(__file__), 'deploy.prototxt')
    model_path = os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel')

    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        print("[ERROR] Face detection model files not found.")
        return

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    # Load Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_yml_path)
    print(f"[INFO] Face recognizer loaded from: {trainer_yml_path}")

    # Load Deep Learning Face Detector
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # Load user names
    with open(names_path, 'r') as f:
        names_data = json.load(f)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Unable to open video file.")
        return

    # Font for display
    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break  # Exit when video ends

        h, w = frame.shape[:2]

        # Convert frame to blob for DNN
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Convert to grayscale and extract face ROI
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y:y1, x:x1]

                # Ensure ROI is valid before resizing
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    face_roi = cv2.resize(face_roi, (200, 200))
                else:
                    continue  # Skip to next face if ROI is invalid

                try:
                    id, conf = recognizer.predict(face_roi)

                    if conf < 50:
                        name = names_data.get(str(id), "Unknown")
                        confidence_text = f"{round(100 - conf)}%"
                    else:
                        name, confidence_text = "Unknown", "N/A"
                except Exception:
                    name, confidence_text = "Unknown", "N/A"

                # Display name & confidence
                cv2.putText(frame, name, (x, y - 10), font, 1, (255, 255, 255), 2)
                cv2.putText(frame, confidence_text, (x, y1 + 20), font, 1, (255, 255, 0), 1)

        # Show video frame
        cv2.imshow('Face Recognition', frame)

        # Press ESC to exit
        if cv2.waitKey(10) & 0xFF == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Example usage: python face_recognition_video.py <path_to_video>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python face_recognition_video.py <path_to_video>")
    else:
        recognize_faces_in_video(sys.argv[1])
