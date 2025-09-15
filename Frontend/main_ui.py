import os
import sys
import threading

import cv2
from PyQt5 import QtWidgets, QtGui, QtSvg, QtCore

from FaceCaptureWorker import FaceCaptureWorker
from FaceRecognizer import FaceRecognizer
from FaceTrainer import FaceTrainer
from VideoUploadPage import VideoUploadPage
from VideoFaceRecognizer import VideoFaceRecognizer
from resources_rc import *


###################################################################################################
# ADMIN LOGIN CLASS
###################################################################################################
class AdminLogin(QtWidgets.QWidget):
    switch_to_dashboard = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("adminLogin")
        self.setWindowTitle("InsightX - Smart Surveillance Login")
        self.setFixedSize(1200, 700)
        self.dark_mode = False
        self.setup_ui()
        self.apply_light_theme()

    def setup_ui(self):
        # Main container
        main_container = QtWidgets.QWidget()
        main_container.setObjectName("loginContainer")
        main_container.setFixedSize(400, 500)

        # Header with toggle
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addStretch()

        # Theme toggle button
        self.theme_toggle = QtWidgets.QPushButton()
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setFixedSize(70, 34)
        self.theme_toggle.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border-radius: 17px;
                border: none;
            }
            QPushButton::indicator {
                width: 30px;
                height: 30px;
                border-radius: 15px;
                background: white;
                margin: 2px;
                image: url(:/icons/sun.svg);
            }
            QPushButton:checked {
                background-color: #4A90E2;
            }
            QPushButton:checked::indicator {
                margin-left: 36px;
                image: url(:/icons/moon.svg);
            }
        """)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_toggle)

        # Main layout
        layout = QtWidgets.QVBoxLayout(main_container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        layout.addLayout(header_layout)

        # Logo and title
        logo_label = QtWidgets.QLabel("InsightX")
        logo_label.setObjectName("logo")
        logo_label.setAlignment(QtCore.Qt.AlignCenter)

        subtitle_label = QtWidgets.QLabel("Facial Recognition System")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(QtCore.Qt.AlignCenter)

        login_title = QtWidgets.QLabel("InsightX – Smart Surveillance Login")
        login_title.setObjectName("login_title")
        login_title.setAlignment(QtCore.Qt.AlignCenter)

        # Input fields
        self.username = QtWidgets.QLineEdit()
        self.username.setPlaceholderText("Username")
        self.username.setObjectName("username")

        self.password = QtWidgets.QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password.setObjectName("password")

        # Login button
        self.login_btn = QtWidgets.QPushButton("Login →")
        self.login_btn.setObjectName("login_btn")
        self.login_btn.clicked.connect(self.check_login)
        self.login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # Footer
        footer_label = QtWidgets.QLabel("Secure access to InsightX dashboard")
        footer_label.setObjectName("footer")
        footer_label.setAlignment(QtCore.Qt.AlignCenter)

        # Add widgets
        layout.addWidget(logo_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(10)
        layout.addWidget(login_title)
        layout.addSpacing(20)
        layout.addWidget(self.username)
        layout.addWidget(self.password)
        layout.addWidget(self.login_btn)
        layout.addWidget(footer_label)
        layout.addStretch()

        # Center container
        outer_layout = QtWidgets.QHBoxLayout(self)
        outer_layout.addStretch()
        outer_layout.addWidget(main_container)
        outer_layout.addStretch()
        self.setLayout(outer_layout)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
        self.style().unpolish(self)
        self.style().polish(self)

    def apply_light_theme(self):
        self.setStyleSheet("""
            QWidget#adminLogin { background-color: #F5F5F5; }
            QWidget#loginContainer { background-color: white; border-radius: 8px; }
            #logo { font-size: 24px; font-weight: bold; color: #333333; margin-bottom: 5px; }
            #subtitle { font-size: 16px; color: #666666; margin-bottom: 15px; }
            #login_title { font-size: 20px; color: #333333; }
            QLineEdit {
                background: white; border: 1px solid #DDDDDD; border-radius: 6px;
                padding: 12px; font-size: 14px; margin-bottom: 15px; height: 45px; color: #333333;
            }
            QLineEdit:focus { border: 2px solid #4A90E2; }
            #login_btn {
                background-color: #4A90E2; color: white; border: none;
                border-radius: 6px; padding: 12px; font-size: 16px; height: 45px;
            }
            #login_btn:hover { background-color: #3A7BC8; }
            #footer { font-size: 12px; color: #999999; margin-top: 15px; }
        """)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget#adminLogin { background-color: #0D0D2B; }
            QWidget#loginContainer { background-color: #1A1A2E; border-radius: 8px; }
            #logo { font-size: 24px; font-weight: bold; color: #FFFFFF; margin-bottom: 5px; }
            #subtitle { font-size: 16px; color: #AAAAAA; margin-bottom: 15px; }
            #login_title { font-size: 20px; color: #FFFFFF; }
            QLineEdit {
                background: #252540; border: 1px solid #7F00FF; border-radius: 6px;
                padding: 12px; font-size: 14px; margin-bottom: 15px; height: 45px; color: #FFFFFF;
            }
            QLineEdit:focus { border: 2px solid #4A90E2; }
            #login_btn {
                background-color: #7F00FF; color: white; border: none;
                border-radius: 6px; padding: 12px; font-size: 16px; height: 45px;
            }
            #login_btn:hover { background-color: #9B30FF; }
            #footer { font-size: 12px; color: #777777; margin-top: 15px; }
        """)

    def check_login(self):
        if self.username.text() == "admin" and self.password.text() == "admin":
            self.switch_to_dashboard.emit()
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid credentials")


###################################################################################################
# DASHBOARD CLASS
###################################################################################################
class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InsightX Dashboard")
        self.setFixedSize(1200, 700)
        self.dark_mode = False
        self.capture = None
        self.timer = None
        self.selected_image_path = None
        self.setup_ui()
        self.apply_light_theme()
        self.face_capture = FaceCaptureWorker()
        self.face_capture.frame_updated.connect(self.update_image_preview)
        self.face_capture.capture_finished.connect(self.capture_done)

    def setup_ui(self):
        # Stacked widget for page navigation
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create pages
        self.create_main_menu()
        self.create_live_feed_page()
        self.create_data_entry_page()
        self.video_upload_page = VideoUploadPage()
        self.stacked_widget.addWidget(self.video_upload_page)
        # self.create_video_upload_page()

    ###########################################################################
    # MAIN MENU PAGE
    ###########################################################################
    def create_main_menu(self):
        # Main menu page
        main_menu = QtWidgets.QWidget()
        main_menu.setObjectName("mainMenu")
        layout = QtWidgets.QVBoxLayout(main_menu)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(40)

        # Header with theme toggle
        header = QtWidgets.QHBoxLayout()

        # Title section
        title_section = QtWidgets.QVBoxLayout()
        title_section.setSpacing(10)

        title = QtWidgets.QLabel("InsightX Dashboard")
        title.setObjectName("dashboardTitle")

        subtitle = QtWidgets.QLabel("Facial Recognition System")
        subtitle.setObjectName("dashboardSubtitle")

        title_section.addWidget(title)
        title_section.addWidget(subtitle)
        header.addLayout(title_section)

        # Add stretch to push toggle to right
        header.addStretch()

        # Theme toggle button
        self.theme_toggle = QtWidgets.QPushButton()
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setFixedSize(70, 34)
        self.theme_toggle.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.theme_toggle.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border-radius: 17px;
                border: none;
            }
            QPushButton::indicator {
                width: 30px;
                height: 30px;
                border-radius: 15px;
                background: white;
                margin: 2px;
                image: url(:/icons/sun.svg);
            }
            QPushButton:checked {
                background-color: #4A90E2;
            }
            QPushButton:checked::indicator {
                margin-left: 36px;
                image: url(:/icons/moon.svg);
            }
        """)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        header.addWidget(self.theme_toggle)

        layout.addLayout(header)

        # Cards container
        cards_container = QtWidgets.QHBoxLayout()
        cards_container.setSpacing(60)

        # Live Feed Card
        live_feed_card = self.create_card(
            "Live Feed",
            "Real-time facial recognition monitoring",
            ":/icons/video_upload.svg",
            "liveFeed"
        )
        live_feed_card.mousePressEvent = lambda _: self.show_page(1)

        # Data Entry Card
        data_entry_card = self.create_card(
            "Data Entry",
            "Register new faces and user information",
            ":/icons/data_entry.svg",
            "dataEntry"
        )
        data_entry_card.mousePressEvent = lambda _: self.show_page(2)

        # Video Upload Card
        video_upload_card = self.create_card(
            "Video Upload",
            "Process recorded videos for face detection",
            ":/icons/live_feed.svg",
            "videoUpload"
        )
        video_upload_card.mousePressEvent = lambda _: self.show_page(3)

        cards_container.addWidget(live_feed_card)
        cards_container.addWidget(data_entry_card)
        cards_container.addWidget(video_upload_card)
        layout.addLayout(cards_container, stretch=1)
        layout.addStretch()

        self.stacked_widget.addWidget(main_menu)

    def create_card(self, title, description, icon_path, object_name):
        card = QtWidgets.QFrame()
        card.setObjectName(f"{object_name}Card")
        card.setFixedSize(320, 400)
        card.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)

        # Card icon with SVG
        icon = QtWidgets.QLabel()
        icon.setObjectName("cardIcon")
        icon.setFixedSize(120, 120)
        icon.setAlignment(QtCore.Qt.AlignCenter)

        # Load SVG icon
        svg_renderer = QtSvg.QSvgRenderer(icon_path)
        svg_pixmap = QtGui.QPixmap(80, 80)
        svg_pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(svg_pixmap)
        svg_renderer.render(painter)
        painter.end()
        icon.setPixmap(svg_pixmap)

        # Card title
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setAlignment(QtCore.Qt.AlignCenter)

        # Card description
        desc_label = QtWidgets.QLabel(description)
        desc_label.setObjectName("cardDescription")
        desc_label.setAlignment(QtCore.Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setFixedWidth(260)

        layout.addWidget(icon, 0, QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()

        return card

    def create_live_feed_page(self):
        self.face_recognizer = FaceRecognizer()

        page = QtWidgets.QWidget()
        page.setObjectName("liveFeedPage")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)

        # Top bar with back button and title
        top_bar = QtWidgets.QHBoxLayout()
        back_btn = self.create_back_button()
        top_bar.addWidget(back_btn)

        title = QtWidgets.QLabel("Live Camera Feed")
        title.setObjectName("pageTitle")
        top_bar.addWidget(title, 1, QtCore.Qt.AlignCenter)

        # Theme toggle
        self.page_theme_toggle = QtWidgets.QPushButton()
        self.page_theme_toggle.setCheckable(True)
        self.page_theme_toggle.setFixedSize(70, 34)
        self.page_theme_toggle.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.page_theme_toggle.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border-radius: 17px;
                border: none;
            }
            QPushButton::indicator {
                width: 30px;
                height: 30px;
                border-radius: 15px;
                background: white;
                margin: 2px;
                image: url(:/icons/sun.svg);
            }
            QPushButton:checked {
                background-color: #4A90E2;
            }
            QPushButton:checked::indicator {
                margin-left: 36px;
                image: url(:/icons/moon.svg);
            }
        """)
        self.page_theme_toggle.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.page_theme_toggle)

        layout.addLayout(top_bar)

        # Main content area - 2:1 ratio
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(20)

        # Video feed section
        video_container = QtWidgets.QFrame()
        video_container.setObjectName("videoContainer")
        video_layout = QtWidgets.QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setObjectName("videoFeed")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Camera loading or not available...")

        # Control buttons
        btn_container = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 10, 0, 0)

        self.start_btn = QtWidgets.QPushButton("Start Camera")
        self.start_btn.setObjectName("actionButton")
        self.stop_btn = QtWidgets.QPushButton("Stop Camera")
        self.stop_btn.setObjectName("actionButton")
        self.stop_btn.setEnabled(False)

        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        video_layout.addWidget(self.video_label)
        video_layout.addWidget(btn_container)

        # Status panel
        status_container = QtWidgets.QFrame()
        status_container.setObjectName("statusContainer")
        status_layout = QtWidgets.QVBoxLayout(status_container)
        status_layout.setContentsMargins(15, 15, 15, 15)
        status_layout.setSpacing(15)

        status_title = QtWidgets.QLabel("System Status")
        status_title.setObjectName("statusTitle")
        status_title.setAlignment(QtCore.Qt.AlignCenter)

        status_items = QtWidgets.QFrame()
        status_items.setObjectName("statusItems")
        items_layout = QtWidgets.QVBoxLayout(status_items)
        items_layout.setContentsMargins(10, 10, 10, 10)
        items_layout.setSpacing(10)

        self.light_status = self.create_status_item("Light Status", "OFF")
        self.fan_status = self.create_status_item("Fan Status", "OFF")
        self.headcount = self.create_status_item("Headcount", "0")

        items_layout.addWidget(self.light_status)
        items_layout.addWidget(self.fan_status)
        items_layout.addWidget(self.headcount)

        person_title = QtWidgets.QLabel("Detected Person")
        person_title.setObjectName("statusTitle")
        person_title.setAlignment(QtCore.Qt.AlignCenter)

        self.person_label = QtWidgets.QLabel("No person detected")
        self.person_label.setObjectName("personLabel")
        self.person_label.setAlignment(QtCore.Qt.AlignCenter)
        self.person_label.setMinimumHeight(100)

        status_layout.addWidget(status_title)
        status_layout.addWidget(status_items)
        status_layout.addSpacing(10)
        status_layout.addWidget(person_title)
        status_layout.addWidget(self.person_label)
        status_layout.addStretch()

        content_layout.addWidget(video_container, 2)
        content_layout.addWidget(status_container, 1)
        layout.addLayout(content_layout, 1)

        # SIGNAL CONNECTIONS
        self.face_recognizer.frame_processed.connect(self.update_video_feed)
        self.face_recognizer.status_updated.connect(self.update_status_info)
        self.face_recognizer.person_detected.connect(self.update_person_info)

        # BUTTON HANDLERS
        def start_camera():
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.camera_thread = threading.Thread(target=self.face_recognizer.start_processing, daemon=True)
            self.camera_thread.start()

        def stop_camera():
            self.face_recognizer.stop_processing()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(start_camera)
        self.stop_btn.clicked.connect(stop_camera)

        self.stacked_widget.addWidget(page)

    @QtCore.pyqtSlot(object)
    def update_video_feed(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    @QtCore.pyqtSlot(dict)
    def update_status_info(self, status):
        self.light_status.findChild(QtWidgets.QLabel, "value").setText(status["light_status"])
        self.fan_status.findChild(QtWidgets.QLabel, "value").setText(status["fan_status"])
        self.headcount.findChild(QtWidgets.QLabel, "value").setText(str(status["headcount"]))

    @QtCore.pyqtSlot(dict)
    def update_person_info(self, person):
        self.person_label.setText(
            f"Name: {person['name']}\n"
            f"Roll No: {person['roll_no']}\n"
            f"Contact: {person['contact']}"
        )

    def create_status_item(self, label, value):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        key = QtWidgets.QLabel(label)
        val = QtWidgets.QLabel(value)
        val.setObjectName("value")
        key.setStyleSheet("font-size: 14px;")
        val.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(key)
        layout.addStretch()
        layout.addWidget(val)
        return frame

    ###########################################################################
    # DATA ENTRY PAGE
    ###########################################################################
    def create_data_entry_page(self):
        page = QtWidgets.QWidget()
        page.setObjectName("dataEntryPage")
        main_layout = QtWidgets.QVBoxLayout(page)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(20)

        # Top bar with back button and title
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 20)

        back_btn = self.create_back_button()
        top_bar.addWidget(back_btn)

        title = QtWidgets.QLabel("Data Entry")
        title.setObjectName("pageTitle")
        top_bar.addWidget(title, 1, QtCore.Qt.AlignCenter)

        # Theme toggle
        self.page_theme_toggle = QtWidgets.QPushButton()
        self.page_theme_toggle.setCheckable(True)
        self.page_theme_toggle.setFixedSize(70, 34)
        self.page_theme_toggle.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.page_theme_toggle.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border-radius: 17px;
                border: none;
            }
            QPushButton::indicator {
                width: 30px;
                height: 30px;
                border-radius: 15px;
                background: white;
                margin: 2px;
                image: url(:/icons/sun.svg);
            }
            QPushButton:checked {
                background-color: #4A90E2;
            }
            QPushButton:checked::indicator {
                margin-left: 36px;
                image: url(:/icons/moon.svg);
            }
        """)
        self.page_theme_toggle.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.page_theme_toggle)

        main_layout.addLayout(top_bar)

        # Main content area (2:1 ratio)
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(30)

        # Left side - Form (2/3 width)
        form_container = QtWidgets.QFrame()
        form_container.setObjectName("formContainer")
        form_layout = QtWidgets.QFormLayout(form_container)
        form_layout.setContentsMargins(30, 30, 30, 30)
        form_layout.setSpacing(20)
        form_layout.setLabelAlignment(QtCore.Qt.AlignLeft)

        # Form fields with dynamic text colors
        full_name_label = QtWidgets.QLabel("Full Name")
        full_name_label.setObjectName("formLabel")
        self.full_name = QtWidgets.QLineEdit()
        self.full_name.setPlaceholderText("Enter person's name")
        self.full_name.setObjectName("formInput")

        roll_no_label = QtWidgets.QLabel("Roll Number")
        roll_no_label.setObjectName("formLabel")
        self.roll_no = QtWidgets.QLineEdit()
        self.roll_no.setPlaceholderText("Enter roll number (e.g. BP-2023)")
        self.roll_no.setObjectName("formInput")

        phone_no_label = QtWidgets.QLabel("Phone Number")
        phone_no_label.setObjectName("formLabel")
        self.phone_no = QtWidgets.QLineEdit()
        self.phone_no.setPlaceholderText("+1 (585) 123-4567")
        self.phone_no.setObjectName("formInput")

        # Add rows to form
        form_layout.addRow(full_name_label, self.full_name)
        form_layout.addRow(roll_no_label, self.roll_no)
        form_layout.addRow(phone_no_label, self.phone_no)

        # Right side - Image capture (1/3 width)
        image_container = QtWidgets.QFrame()
        image_container.setObjectName("imageContainer")
        image_layout = QtWidgets.QVBoxLayout(image_container)
        image_layout.setContentsMargins(20, 20, 20, 20)
        image_layout.setSpacing(15)

        # Image preview with dynamic colors
        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setMinimumSize(200, 200)
        self.image_preview.setText("Upload a clear face image")
        self.image_preview.setObjectName("imagePreview")

        # Camera/Upload buttons
        btn_container = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(10)

        self.capture_btn = QtWidgets.QPushButton("Take Photo")
        self.capture_btn.setObjectName("actionButton")
        self.capture_btn.clicked.connect(self.capture_image)

        self.select_image_btn = QtWidgets.QPushButton("Upload Image")
        self.select_image_btn.setObjectName("actionButton")
        self.select_image_btn.clicked.connect(self.select_image)

        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.select_image_btn)

        image_layout.addWidget(self.image_preview)
        image_layout.addWidget(btn_container)
        image_layout.addStretch()

        # Add both containers to main content
        content_layout.addWidget(form_container, 2)  # 2/3 width
        content_layout.addWidget(image_container, 1)  # 1/3 width

        main_layout.addWidget(content_widget)

        # Register button at bottom center
        btn_widget = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 20, 0, 0)

        self.register_btn = QtWidgets.QPushButton("Register Person")
        self.register_btn.setObjectName("registerButton")
        self.register_btn.setFixedWidth(200)
        self.register_btn.clicked.connect(self.register_person)
        btn_layout.addWidget(self.register_btn, 0, QtCore.Qt.AlignCenter)

        main_layout.addWidget(btn_widget)
        main_layout.addStretch()

        self.stacked_widget.addWidget(page)

        # Apply initial theme styling
        self.update_data_entry_theme()

    def update_data_entry_theme(self):
        """Update the data entry page colors based on current theme"""
        if self.dark_mode:  # Dark mode
            text_color = "#FFFFFF"  # White text
            placeholder_color = "#AAAAAA"  # Light gray placeholder
            border_color = "#7F00FF"  # Purple border for dark mode
            bg_color = "#252540"  # Dark background
        else:  # Light mode
            text_color = "#000000"  # Black text
            placeholder_color = "#777777"  # Dark gray placeholder
            border_color = "#AAAAAA"  # Light border for light mode
            bg_color = "#F8F8F8"  # Light background

        # Apply the styles
        self.setStyleSheet(f"""
            /* Form labels */
            QLabel#formLabel {{
                color: {text_color};
                font-size: 14px;
            }}

            /* Image preview */
            QLabel#imagePreview {{
                color: {placeholder_color};
                border: 2px dashed {border_color};
                border-radius: 5px;
                background-color: {bg_color};
                font-size: 14px;
            }}

            /* Form inputs */
            QLineEdit#formInput {{
                color: {text_color};
                background-color: {'#1A1A2E' if self.dark_mode else '#FFFFFF'};
                border: 1px solid {border_color};
            }}

            /* Page title */
            QLabel#pageTitle {{
                color: {'#7F00FF' if self.dark_mode else '#4A90E2'};
            }}
        """)

    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
        # Update all toggle buttons
        self.theme_toggle.setChecked(self.dark_mode)
        if hasattr(self, 'page_theme_toggle'):
            self.page_theme_toggle.setChecked(self.dark_mode)
        # Update data entry page specifically
        self.update_data_entry_theme()

    def update_image_preview(self, image: QtGui.QImage):
        self.image_preview.setPixmap(QtGui.QPixmap.fromImage(image))

    def capture_done(self):
        QtWidgets.QMessageBox.information(self, "Capture Complete", "30 face images captured successfully.")

    def capture_image(self):
        name = self.full_name.text().strip()
        roll = self.roll_no.text().strip()
        contact = self.phone_no.text().strip()

        if not all([name, roll, contact]):
            QtWidgets.QMessageBox.warning(self, "Incomplete Info", "Please fill all fields before taking a photo.")
            return

        self.face_capture.start_capture(name, roll, contact)


    def select_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Face Image", "",
            "Image Files (*.png *.jpg *.jpeg)", options=options
        )
        if file_path:
            self.selected_image_path = file_path
            pixmap = QtGui.QPixmap(file_path)
            if not pixmap.isNull():
                self.image_preview.setPixmap(
                    pixmap.scaled(self.image_preview.width(), self.image_preview.height(),
                                  QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                )
                self.image_preview.setStyleSheet("border: none;")

    def register_person(self):
        name = self.full_name.text().strip()
        roll = self.roll_no.text().strip()
        contact = self.phone_no.text().strip()

        # Basic validation
        if not all([name, roll]):
            QtWidgets.QMessageBox.warning(self, "Error", "Full Name and Roll Number are required")
            return

        # Check if either an image was uploaded OR a face was captured
        captured_dir = "images"
        face_id_prefix = f"User-{roll}"
        has_captured_images = any(
            fname.startswith(f"User-") and roll in fname for fname in os.listdir(captured_dir)) if os.path.exists(
            captured_dir) else False

        if not hasattr(self, 'selected_image_path') and not has_captured_images:
            QtWidgets.QMessageBox.warning(self, "Error", "Please capture or upload an image")
            return

        # Create a trainer instance and train
        trainer = FaceTrainer()
        trainer.train()

        QtWidgets.QMessageBox.information(self, "Success", "Person registered successfully")
        self.clear_form()

    def clear_form(self):
        self.full_name.clear()
        self.roll_no.clear()
        self.phone_no.clear()
        self.image_preview.clear()
        self.image_preview.setText("Upload a clear face image")

        if hasattr(self, 'selected_image_path'):
            del self.selected_image_path

    ###########################################################################
    # VIDEO UPLOAD PAGE
    ###########################################################################
    def create_video_upload_page(self):
        page = QtWidgets.QWidget()
        page.setObjectName("videoUploadPage")
        main_layout = QtWidgets.QVBoxLayout(page)
        main_layout.setContentsMargins(60, 40, 60, 60)
        main_layout.setSpacing(30)

        # Header Section
        header_layout = QtWidgets.QVBoxLayout()
        header_layout.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addLayout(header_layout)

        # Control Bar
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.setContentsMargins(0, 20, 0, 20)

        back_btn = self.create_back_button()
        control_bar.addWidget(back_btn)

        page_title = QtWidgets.QLabel("Video Upload")
        page_title.setObjectName("pageTitle")
        control_bar.addWidget(page_title, 1, QtCore.Qt.AlignCenter)

        self.page_theme_toggle = QtWidgets.QPushButton()
        self.page_theme_toggle.setCheckable(True)
        self.page_theme_toggle.setFixedSize(70, 34)
        self.page_theme_toggle.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.page_theme_toggle.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0;
                border-radius: 17px;
                border: none;
            }
            QPushButton::indicator {
                width: 30px;
                height: 30px;
                border-radius: 15px;
                background: white;
                margin: 2px;
                image: url(:/icons/sun.svg);
            }
            QPushButton:checked {
                background-color: #4A90E2;
            }
            QPushButton:checked::indicator {
                margin-left: 36px;
                image: url(:/icons/moon.svg);
            }
        """)
        self.page_theme_toggle.clicked.connect(self.toggle_theme)
        control_bar.addWidget(self.page_theme_toggle)

        main_layout.addLayout(control_bar)

        # Main Content - Using Card Styling
        content_card = QtWidgets.QFrame()
        content_card.setObjectName("videoUploadCard")  # Using card class
        content_layout = QtWidgets.QVBoxLayout(content_card)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(30)

        # Drop Zone - Styled as Card
        drop_card = QtWidgets.QFrame()
        drop_card.setObjectName("videoUploadCard")  # Using card class
        drop_card.setMinimumSize(700, 300)
        drop_card.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        drop_layout = QtWidgets.QVBoxLayout(drop_card)
        drop_layout.setContentsMargins(50, 50, 50, 50)
        drop_layout.setSpacing(20)
        drop_layout.setAlignment(QtCore.Qt.AlignCenter)

        # Upload Icon - Using cardIcon class
        upload_icon = QtWidgets.QLabel()
        upload_icon.setPixmap(QtGui.QPixmap(r"C:\Users\hamda\OneDrive\Documents\GitHub\INSIGHT_X\Frontend\resource\icons\upload.svg").scaled(80, 80,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        upload_icon.setObjectName("cardIcon")
        upload_icon.setAlignment(QtCore.Qt.AlignCenter)
        drop_layout.addWidget(upload_icon)

        # Instruction Text - Using cardDescription class
        instruction_text = QtWidgets.QLabel(
            "Drag and drop video file\n"
            "Upload a video to detect and recognize faces"
        )
        instruction_text.setObjectName("cardDescription")
        instruction_text.setAlignment(QtCore.Qt.AlignCenter)
        instruction_text.setWordWrap(True)
        drop_layout.addWidget(instruction_text)

        content_layout.addWidget(drop_card, 0, QtCore.Qt.AlignCenter)

        # Select Button - Using actionButton class
        select_btn = QtWidgets.QPushButton("Select Video")
        select_btn.setObjectName("actionButton")
        select_btn.setFixedSize(220, 50)
        select_btn.clicked.connect(self.select_video)
        content_layout.addWidget(select_btn, 0, QtCore.Qt.AlignCenter)

        main_layout.addWidget(content_card)
        self.stacked_widget.addWidget(page)
        self.update_video_upload_theme()

    def update_video_upload_theme(self):
        """Theme updates handled by main apply_light_theme/apply_dark_theme methods"""
        # No additional styling needed - using existing card classes
        self.style().unpolish(self)
        self.style().polish(self)

    def select_video(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)", options=options
        )
        if file_path:
            self.video_path = file_path
            recognizer = VideoFaceRecognizer(parent=self)
            recognizer.recognize(self.video_path)

    ###########################################################################
    # UTILITY METHODS
    ###########################################################################
    def create_back_button(self):
        btn = QtWidgets.QPushButton("← Back to Dashboard")
        btn.setObjectName("backButton")
        btn.clicked.connect(lambda: self.show_page(0))
        return btn

    def show_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        # Sync theme toggle state
        if hasattr(self, 'page_theme_toggle'):
            self.page_theme_toggle.setChecked(self.dark_mode)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
        # Update all toggle buttons
        self.theme_toggle.setChecked(self.dark_mode)
        if hasattr(self, 'page_theme_toggle'):
            self.page_theme_toggle.setChecked(self.dark_mode)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                                   QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

    def stop_camera(self):
        self.timer.stop()
        self.capture.release()
        self.video_label.clear()
        self.video_label.setText("Camera loading or not available...")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    def apply_light_theme(self):
        self.setStyleSheet("""
            /* Main window */
            QWidget {
                background-color: #F5F5F5;
                font-family: 'Segoe UI';
            }

            /* Main menu */
            #mainMenu {
                background-color: #F5F5F5;
            }

            /* Dashboard title */
            #dashboardTitle {
                font-size: 36px;
                font-weight: bold;
                color: #333333;
                margin-bottom: 5px;
            }

            #dashboardSubtitle {
                font-size: 20px;
                color: #666666;
            }

            /* Cards */
            #liveFeedCard, #dataEntryCard, #videoUploadCard {
                background-color: #FFFFFF;
                border-radius: 16px;
                border: none;
            }

            #liveFeedCard:hover, #dataEntryCard:hover, #videoUploadCard:hover {
                background-color: #F0F0F0;
            }

            #cardTitle {
                font-size: 24px;
                font-weight: bold;
                color: #4A90E2;
                margin-top: 10px;
            }

            #cardDescription {
                font-size: 16px;
                color: #666666;
                line-height: 1.4;
            }

            #cardIcon {
                background-color: #F0F0F0;
                border-radius: 60px;
                border: 2px solid #4A90E2;
                padding: 10px;
            }

            /* Pages */
            #liveFeedPage, #dataEntryPage, #videoUploadPage {
                background-color: #F5F5F5;
            }

            #pageTitle {
                font-size: 32px;
                font-weight: bold;
                color: #4A90E2;
                margin-bottom: 30px;
            }

            /* Back button */
            #backButton {
                background: transparent;
                color: #4A90E2;
                border: none;
                font-size: 14px;
                padding: 5px;
                text-align: left;
            }

            #backButton:hover {
                color: #3A7BC8;
            }

            /* Form elements */
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #DDDDDD;
                border-radius: 6px;
                padding: 12px;
                color: #333333;
                font-size: 14px;
            }

            QLineEdit:focus {
                border: 1px solid #4A90E2;
            }

            #actionButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 16px;
            }

            #actionButton:hover {
                background-color: #3A7BC8;
            }

            /* Video feed */
            #videoFeed {
                background-color: #F5F5F5;
                border: 2px solid #DDDDDD;
                border-radius: 4px;
            }

            /* Image Preview */
            #imagePreview {
                background-color: #F0F0F0;
                border: 2px dashed #CCCCCC;
                border-radius: 4px;
                color: #666666;
                font-size: 14px;
            }

            /* Status Panel */
            #statusContainer {
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #DDDDDD;
            }
            #statusItems {
                background-color: #F5F5F5;
                border-radius: 6px;
                border: 1px solid #DDDDDD;
            }
            #statusTitle {
                font-size: 18px;
                font-weight: bold;
                color: #4A90E2;
            }
            #statusLabel {
                font-size: 14px;
                color: #666666;
            }
            #statusValue {
                font-size: 14px;
                font-weight: bold;
                color: #333333;
            }
            #personLabel {
                background-color: #F5F5F5;
                border-radius: 6px;
                border: 1px solid #DDDDDD;
                padding: 10px;
            }
        """)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            /* Main window */
            QWidget {
                background-color: #0D0D2B;
                font-family: 'Segoe UI';
            }

            /* Main menu */
            #mainMenu {
                background-color: #0D0D2B;
            }

            /* Dashboard title */
            #dashboardTitle {
                font-size: 36px;
                font-weight: bold;
                color: #FFFFFF;
                margin-bottom: 5px;
            }

            #dashboardSubtitle {
                font-size: 20px;
                color: #AAAAAA;
            }

            /* Cards */
            #liveFeedCard, #dataEntryCard, #videoUploadCard {
                background-color: #1A1A2E;
                border-radius: 16px;
                border: none;
            }

            #liveFeedCard:hover, #dataEntryCard:hover, #videoUploadCard:hover {
                background-color: #252540;
            }

            #cardTitle {
                font-size: 24px;
                font-weight: bold;
                color: #7F00FF;
                margin-top: 10px;
            }

            #cardDescription {
                font-size: 16px;
                color: #CCCCCC;
                line-height: 1.4;
            }

            #cardIcon {
                background-color: #252540;
                border-radius: 60px;
                border: 2px solid #7F00FF;
                padding: 10px;
            }

            /* Pages */
            #liveFeedPage, #dataEntryPage, #videoUploadPage {
                background-color: #0D0D2B;
            }

            #pageTitle {
                font-size: 32px;
                font-weight: bold;
                color: #7F00FF;
                margin-bottom: 30px;
            }

            /* Back button */
            #backButton {
                background: transparent;
                color: #7F00FF;
                border: none;
                font-size: 14px;
                padding: 5px;
                text-align: left;
            }

            #backButton:hover {
                color: #9B30FF;
            }

            /* Form elements */
            QLineEdit {
                background-color: #1A1A2E;
                border: 1px solid #7F00FF;
                border-radius: 6px;
                padding: 12px;
                color: white;
                font-size: 14px;
            }

            QLineEdit:focus {
                border: 1px solid #9B30FF;
            }

            #actionButton {
                background-color: #7F00FF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 16px;
            }

            #actionButton:hover {
                background-color: #9B30FF;
            }

            /* Video feed */
            #videoFeed {
                background-color: #0D0D2B;
                border: 2px solid #7F00FF;
                border-radius: 4px;
            }

            /* Image Preview */
            #imagePreview {
                background-color: #252540;
                border: 2px dashed #7F00FF;
                border-radius: 4px;
                color: #AAAAAA;
                font-size: 14px;
            }

            /* Status Panel */
            #statusContainer {
                background-color: #1A1A2E;
                border-radius: 8px;
                border: 1px solid #7F00FF;
            }
            #statusItems {
                background-color: #252540;
                border-radius: 6px;
                border: 1px solid #7F00FF;
            }
            #statusTitle {
                font-size: 18px;
                font-weight: bold;
                color: #7F00FF;
            }
            #statusLabel {
                font-size: 14px;
                color: #CCCCCC;
            }
            #statusValue {
                font-size: 14px;
                font-weight: bold;
                color: #FFFFFF;
            }
            #personLabel {
                background-color: #252540;
                border-radius: 6px;
                border: 1px solid #7F00FF;
                padding: 10px;
            }
        """)


###################################################################################################
# MAIN APPLICATION CLASS
###################################################################################################
class MainApp(QtWidgets.QStackedWidget):
    def __init__(self):
        super().__init__()
        self.login = AdminLogin()
        self.dashboard = Dashboard()
        self.addWidget(self.login)
        self.addWidget(self.dashboard)
        self.login.switch_to_dashboard.connect(lambda: self.setCurrentWidget(self.dashboard))


###################################################################################################
# APPLICATION ENTRY POINT
###################################################################################################
def run_app():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.setFixedSize(1200, 700)
    window.setWindowTitle("InsightX")
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_app()