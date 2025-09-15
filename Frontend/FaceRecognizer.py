import json
import os
import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# Optional audio dependencies
try:
    import pyaudio  # type: ignore
    import scipy.signal  # type: ignore
    _AUDIO_AVAILABLE = True
except Exception:
    _AUDIO_AVAILABLE = False


class FaceRecognizer(QObject):
    # Signals to communicate with UI thread
    person_detected = pyqtSignal(dict)  # {name, roll_no, contact}
    status_updated = pyqtSignal(dict)  # {light_status, fan_status, headcount}
    frame_processed = pyqtSignal(object)  # Processed frame (for display)

    def __init__(self):
        super().__init__()
        self.running = False
        # Resolve repo paths relative to this file so it works from any CWD
        self._repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._models_dir = os.path.join(self._repo_root, 'real-time-face-recognition')
        self._trainer_path = os.path.join(self._repo_root, 'trainer.yml')
        self._names_path = os.path.join(self._repo_root, 'names.json')

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(self._trainer_path):
            self.recognizer.read(self._trainer_path)
        else:
            # Leave uninitialized; predictions will be skipped
            self.recognizer = None  # type: ignore

        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(self._models_dir, 'deploy.prototxt'),
            os.path.join(self._models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        )
        self.users = self._load_users()
        self.current_status = {
            'light_status': "OFF",
            'fan_status': "OFF",
            'headcount': 0
        }

        # Start fan detection thread only if audio deps are available
        if _AUDIO_AVAILABLE:
            self.fan_thread = threading.Thread(target=self.detect_fan_status, daemon=True)
            self.fan_thread.start()

    def _load_users(self):
        try:
            with open(self._names_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def detect_fan_status(self, threshold=6000, smoothing_factor=5):
        if not _AUDIO_AVAILABLE:
            return
        CHUNK = 2048
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        recent_amplitudes = []

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                filtered_data = scipy.signal.lfilter(
                    *scipy.signal.butter(5, [50 / (0.5 * 44100), 300 / (0.5 * 44100)], btype='band'), audio_data)
                fft_data = np.abs(np.fft.fft(filtered_data))
                max_amplitude = np.max(fft_data)

                recent_amplitudes.append(max_amplitude)
                if len(recent_amplitudes) > smoothing_factor:
                    recent_amplitudes.pop(0)
                avg_amplitude = np.mean(recent_amplitudes)

                new_status = "ON" if avg_amplitude > threshold else "OFF"
                if new_status != self.current_status['fan_status']:
                    self.current_status['fan_status'] = new_status
                    self.status_updated.emit(self.current_status.copy())

                time.sleep(0.1)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def process_frame(self, frame):
        # Light detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        light_status = "ON" if brightness > 80 else "OFF"

        if light_status != self.current_status['light_status']:
            self.current_status['light_status'] = light_status
            self.status_updated.emit(self.current_status.copy())

        # Face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        current_person = {"name": "Unknown", "roll_no": "N/A", "contact": "N/A"}

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                (h, w) = frame.shape[:2]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append((x, y, x1, y1))

                # Face recognition
                face_roi = gray[y:y1, x:x1]
                # Only try recognition if a trained model is available
                if self.recognizer is not None and face_roi.size > 0:
                    try:
                        face_roi_resized = cv2.resize(face_roi, (200, 200))
                        id, confidence = self.recognizer.predict(face_roi_resized)
                        if confidence < 50:
                            current_person = self.users.get(str(id), {"name": "Unknown", "roll_no": "N/A", "contact": "N/A"})
                    except Exception:
                        pass

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Update headcount
        if len(faces) != self.current_status['headcount']:
            self.current_status['headcount'] = len(faces)
            self.status_updated.emit(self.current_status.copy())

        # Emit signals
        self.person_detected.emit(current_person)
        self.frame_processed.emit(frame)

        return frame

    def start_processing(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
            time.sleep(0.03)  # ~30fps

    def stop_processing(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()