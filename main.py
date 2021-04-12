from QrDetector import QrDetector
import cv2


def main():
    detector = QrDetector()
    while True:
        detector.detect()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
