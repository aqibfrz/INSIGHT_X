import cv2
import numpy as np
import json
import os
import pyaudio
import threading
import scipy.signal
import time


def detect_light_status(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    light_threshold = 80
    return "Lights ON" if brightness > light_threshold else "Lights OFF"


fan_status = "Fan OFF"
last_fan_status = "Fan OFF"
fan_status_lock = threading.Lock()

 
def detect_fan_status(threshold=6000, smoothing_factor=5):

    global fan_status, last_fan_status
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    recent_amplitudes = []

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    def bandpass_filter(data, lowcut=50, highcut=300, fs=44100, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return scipy.signal.lfilter(b, a, data)

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Apply bandpass filter to focus on fan noise
            filtered_data = bandpass_filter(audio_data)

            # Compute frequency spectrum
            fft_data = np.abs(np.fft.fft(filtered_data))
            max_amplitude = np.max(fft_data)

            # Apply moving average smoothing
            recent_amplitudes.append(max_amplitude)
            if len(recent_amplitudes) > smoothing_factor:
                recent_amplitudes.pop(0)
            avg_amplitude = np.mean(recent_amplitudes)

            # Determine fan status
            new_fan_status = "Fan ON" if avg_amplitude > threshold else "Fan OFF"

            # Avoid rapid switching between ON/OFF
            with fan_status_lock:
                if new_fan_status != last_fan_status:
                    time.sleep(1)
                    fan_status = new_fan_status
                    last_fan_status = new_fan_status

    except Exception as e:
        print(f"[ERROR] Fan detection error: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":

    # Start fan detection in a separate thread
    fan_thread = threading.Thread(target=detect_fan_status, daemon=True)
    fan_thread.start()

    # Create LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    # Load Deep Learning Face Detector
    net = cv2.dnn.readNetFromCaffe(".\\real-time-face-recognition\\deploy.prototxt",
                                   ".\\real-time-face-recognition\\res10_300x300_ssd_iter_140000.caffemodel")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load user names from JSON
    names = ['Unknown']
    try:
        with open('names.json', 'r') as fs:
            users = json.load(fs)
    except Exception as e:
        print(f"[ERROR] Unable to load names.json: {e}")

    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)

    while True:
        ret, img = cam.read()
        h, w = img.shape[:2]

        light_status = detect_light_status(img)

        # Convert to blob for DNN model
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y:y1, x:x1]

                try:
                    id, confidence = recognizer.predict(face_roi)
                    if confidence < 50:
                        user = users.get(str(id), {"name": "Unknown", "roll_no": "N/A", "contact": "N/A"})
                        name = f"Name: {user['name']}"
                        roll_no = f"Roll No: {user['roll_no']}"
                        contact = f"Contact: {user['contact']}"
                        confidence_text = f"Confidence: {round(100 - confidence)}%"
                    else:
                        name, roll_no, contact, confidence_text = "Unknown", "N/A", "N/A", "N/A"
                except Exception as e:
                    print(e)
                    name, roll_no, contact, confidence_text = "Unknown", "N/A", "N/A", "N/A"

                height, width, _ = img.shape
                x_offset = 10
                y_offset = height - 80

                cv2.putText(img, name, (x_offset, y_offset), font, 0.8, (255, 255, 255), 2)
                cv2.putText(img, roll_no, (x_offset, y_offset + 25), font, 0.8, (255, 255, 255), 2)
                cv2.putText(img, contact, (x_offset, y_offset + 50), font, 0.8, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x_offset, y_offset + 75), font, 0.8, (255, 255, 0), 1)

        # Display light and fan status
        cv2.putText(img, light_status, (50, 50), font, 1, (0, 255, 255), 2)

        with fan_status_lock:
            cv2.putText(img, fan_status, (50, 100), font, 1, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', img)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    print("\n[INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()
