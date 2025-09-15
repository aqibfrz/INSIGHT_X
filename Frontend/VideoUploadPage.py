import os

from PyQt5 import QtWidgets, QtCore

from VideoFaceRecognizer import VideoFaceRecognizer


class VideoUploadPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Upload and Face Recognition")
        self.setStyleSheet("background-color: #1e1e1e; color: white; font-family: Arial;")
        self.video_path = None

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Upload a Video for Face Recognition")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)

        self.drag_drop_area = DragDropArea(self)
        self.drag_drop_area.file_dropped.connect(self.process_video)

        or_label = QtWidgets.QLabel("OR")
        or_label.setAlignment(QtCore.Qt.AlignCenter)
        or_label.setStyleSheet("font-size: 16px;")

        upload_button = QtWidgets.QPushButton("Select Video File")
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
        """)
        upload_button.clicked.connect(self.select_video)

        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(self.drag_drop_area)
        layout.addSpacing(20)
        layout.addWidget(or_label)
        layout.addSpacing(10)
        layout.addWidget(upload_button, alignment=QtCore.Qt.AlignCenter)

    def select_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self.process_video(file_path)

    def process_video(self, file_path):
        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.critical(self, "Error", "Selected file does not exist.")
            return

        self.video_path = file_path
        recognizer = VideoFaceRecognizer(parent=self)
        recognizer.recognize(file_path)


class DragDropArea(QtWidgets.QLabel):
    file_dropped = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Drag and drop a video file here")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #0078d7;
                font-size: 16px;
                padding: 40px;
                color: #aaa;
            }
            QLabel:hover {
                border-color: #005fa3;
                color: #ddd;
            }
        """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.file_dropped.emit(file_path)
            else:
                QtWidgets.QMessageBox.warning(self, "Invalid File", "Please drop a valid video file.")
