import cv2 
from src.face_detector import FaceDetector

def main():
    face_detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("gg")
        return
    while True:

        ok, frame = cap.read()
        if not ok:
            break
        # gray = cv2.
        faces = face_detector.detect(frame)
        for (x,y,z,w) in faces:
            cv2.rectangle(frame, (x, y), (x+w,y+z), (0,255,0), 2)
        cv2.imshow("face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()