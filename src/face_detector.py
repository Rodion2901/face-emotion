import cv2

class FaceDetector:
    def __init__(self):
        """
        i take data and with cascadeclassifier
        and do something
        """
        self.path_cascade = "data/haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(self.path_cascade)
    def detect(self, frame):
        """
         i take video and chenge from bgr ot gray
         and with data i return x y z w where z y are 
         center face and z w are height and width
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (48,48),
        )
        return faces