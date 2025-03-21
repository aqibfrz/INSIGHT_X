import cv2

class Video(object):
    net = cv2.dnn.readNetFromCaffe(".\\real-time-face-recognition\\deploy.prototxt", ".\\real-time-face-recognition\\res10_300x300_ssd_iter_140000.caffemodel")
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __init__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()        