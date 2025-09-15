import os
import json
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui


class FaceCaptureWorker(QtCore.QObject):
    frame_updated = QtCore.pyqtSignal(QtGui.QImage)
    capture_finished = QtCore.pyqtSignal()

    def __init__(self, preview_label: QtGui.QImage = None, target_count=30, image_dir='images', json_file='names.json'):
        super().__init__()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.capture = None
        self.net = None

        self.face_id = None
        self.face_name = ""
        self.roll_no = ""
        self.contact = ""

        self.target_count = target_count
        self.frame_count = 0

        # Resolve repo root for consistent IO
        self._repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._models_dir = os.path.join(self._repo_root, 'real-time-face-recognition')
        self.image_dir = os.path.join(self._repo_root, image_dir)
        self.json_file = os.path.join(self._repo_root, json_file)

        self.preview_label = preview_label  # Optional QLabel reference (not used directly here)

        self.load_model()

    def load_model(self):
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(self._models_dir, 'deploy.prototxt'),
            os.path.join(self._models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        )

    def start_capture(self, name: str, roll: str, contact: str):
        self.face_name = name
        self.roll_no = roll
        self.contact = contact

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.face_id = self.get_next_face_id()
        self.save_name_json()

        self.capture = cv2.VideoCapture(0)
        self.frame_count = 0
        self.timer.start(30)

    def stop_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self.timer.stop()
        self.capture_finished.emit()

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_crop = frame[y:y1, x:x1]
                self.frame_count += 1
                cv2.imwrite(f"{self.image_dir}/User-{self.face_id}-{self.frame_count}.jpg", face_crop)
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QtGui.QImage.Format_RGB888)
        self.frame_updated.emit(qt_image)

        if self.frame_count >= self.target_count:
            self.stop_capture()

    def get_next_face_id(self):
        user_ids = set()
        for filename in os.listdir(self.image_dir):
            try:
                number = int(filename.split("-")[1])
                user_ids.add(number)
            except (IndexError, ValueError):
                continue
        return min(set(range(1, max(user_ids, default=0) + 2)) - user_ids)

    def save_name_json(self):
        data = {}
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass

        data[str(self.face_id)] = {
            "name": self.face_name,
            "roll_no": self.roll_no,
            "contact": self.contact
        }

        with open(self.json_file, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
