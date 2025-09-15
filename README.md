## InsightX – Face Recognition using OpenCV

Modern PyQt5 desktop app for face registration and recognition with OpenCV DNN face detector and LBPH recognizer.

### Features
- **Dashboard UI**: Light/Dark themes, navigation cards
- **Data Entry**: Capture faces from webcam or upload image; auto-saves to `images/` and `names.json`
- **Training**: Train LBPH recognizer; saves model to `trainer.yml`
- **Live Feed**: Real‑time face detection/recognition, headcount, light/fan status
- **Video Upload**: Run recognition on recorded videos
- **Robust paths**: Works regardless of current working directory

### Project Structure
```
face-recognition-using-open-cv/
  Frontend/
    main_ui.py                # App entry (PyQt5 UI)
    FaceCaptureWorker.py      # Webcam capture and save faces
    FaceRecognizer.py         # Live recognition pipeline
    FaceTrainer.py            # Training pipeline (LBPH)
    VideoUploadPage.py        # Drag/drop video UI
    VideoFaceRecognizer.py    # Recognition on videos
    resource/…                # Icons and Qt resources
  real-time-face-recognition/
    deploy.prototxt
    res10_300x300_ssd_iter_140000.caffemodel
    face_recognition_video.py # CLI example for video
    face_recognizer.py        # Alt recognizer module (for reference)
    face_taker.py, face_train.py (legacy scripts)
  images/                     # Auto‑created; captured face crops
  names.json                  # Auto‑created mapping of id → {name, roll_no, contact}
  trainer.yml                 # Auto‑created trained model
  requirements.txt
```

### Requirements
- Windows 10/11, Python 3.10–3.11
- Webcam for live capture
- Packages (installed via `requirements.txt`):
  - `opencv-contrib-python` (includes LBPH)
  - `PyQt5`, `numpy`, `Pillow`, `imutils`, `scipy`
  - `pyaudio` (optional; fan status; app works without it)

### Setup
```powershell
cd face-recognition-using-open-cv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `pyaudio` fails to install on your system, you can skip it (fan detection will auto‑disable):
```powershell
pip install --only-binary=:all: pyaudio==0.2.13 || echo "Skipping PyAudio"
```

### Run the App
```powershell
python Frontend\main_ui.py
```
- Login with: `admin` / `admin`

### Usage Flow
1) **Register a person**
   - Go to Data Entry
   - Enter Full Name, Roll Number, Contact
   - Click "Take Photo" (captures ~30 images) or "Upload Image"
   - Click "Register Person" to train
   - Generates/updates: `images/`, `names.json`, `trainer.yml`

2) **Live recognition**
   - Go to Live Feed → Start Camera
   - Shows headcount and recognized person details

3) **Video recognition**
   - Go to Video Upload → drag a video file or click Select Video
   - Press ESC to close the preview window

### Models
Pretrained OpenCV face detector files are included:
- `real-time-face-recognition/deploy.prototxt`
- `real-time-face-recognition/res10_300x300_ssd_iter_140000.caffemodel`

The trained recognizer model is produced after registration/training:
- `trainer.yml` at project root (shared by all components)

### Troubleshooting
- **ModuleNotFoundError: cv2**
  - Ensure the venv is active and run: `pip install opencv-contrib-python==4.8.1.78`
  - If issues persist, try: `4.7.0.72`

- **Pillow build error on Python 3.11**
  - The project pins `Pillow==10.4.0`. Run `pip install -r requirements.txt` after upgrading `pip`.

- **Camera not opening**
  - Close other apps using the webcam. Try a different index: change `cv2.VideoCapture(0)` → `(1)` if needed.

- **No recognition results**
  - Ensure you trained at least one identity: Data Entry → Take Photo → Register Person
  - Check that `images/` has face crops; `trainer.yml` exists; `names.json` has your details

- **Fan status not updating**
  - Install `pyaudio` and `scipy`. If unavailable, the app disables fan detection automatically.

### Development Notes
- All file IO uses repo‑relative paths so you can run from anywhere inside the project
- `FaceTrainer` saves `trainer.yml` to project root; `FaceRecognizer`/`VideoFaceRecognizer` read from there

### License
This project is provided as‑is for educational purposes.


