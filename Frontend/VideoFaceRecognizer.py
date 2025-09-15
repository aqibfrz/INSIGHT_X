import cv2
import numpy as np
import json
import os
from PyQt5.QtWidgets import QMessageBox

class VideoFaceRecognizer:
    def __init__(self, parent=None):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(repo_root, 'real-time-face-recognition')
        self.trainer_yml_path = os.path.join(repo_root, 'trainer.yml')
        self.names_path = os.path.join(repo_root, 'names.json')
        self.proto_path = os.path.join(models_dir, 'deploy.prototxt')
        self.model_path = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.parent = parent

        try:
            self._check_files()
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.trainer_yml_path)
            self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
            with open(self.names_path, 'r') as f:
                self.names_data = json.load(f)
        except Exception as e:
            self._show_error(str(e))

    def _check_files(self):
        for path in [self.trainer_yml_path, self.names_path, self.proto_path, self.model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

    def _show_error(self, message):
        if self.parent:
            QMessageBox.critical(self.parent, "Error", message)
        else:
            print("[ERROR]", message)

    def recognize(self, video_path):
        if not os.path.exists(video_path):
            self._show_error(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._show_error("Unable to open video file.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_roi = gray[y:y1, x:x1]
                    if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                        continue

                    face_roi = cv2.resize(face_roi, (200, 200))

                    try:
                        id, conf = self.recognizer.predict(face_roi)
                        if conf < 50:
                            name = self.names_data.get(str(id), {}).get("name", "Unknown")
                            confidence_text = f"{round(100 - conf)}%"
                        else:
                            name, confidence_text = "Unknown", "N/A"
                    except:
                        name, confidence_text = "Unknown", "N/A"

                    cv2.putText(frame, name, (x, y - 10), self.font, 1, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x, y1 + 20), self.font, 1, (255, 255, 0), 1)

            cv2.imshow('Face Recognition - Press ESC to Exit', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
