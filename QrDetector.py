import cv2
import numpy as np
from util import cv_functions as cf
from util import math_functions as mf
from util import printer


class QrDetector:

    def __init__(self, webcam=0, width=1280, height=720, detection_frame=250, threshold=217):
        self.width = width
        self.height = height
        self.detection_frame = detection_frame
        self.threshold = threshold
        self.cap = cv2.VideoCapture(webcam)

    def apply_threshold(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, self.threshold, 255, cv2.THRESH_BINARY)
        return img_thresh

    def detect(self):
        _, img = self.cap.read()
        img = cv2.resize(img, (self.width, self.height))
        frame = np.zeros((self.detection_frame, self.detection_frame))

        img_thresh = self.apply_threshold(img)
        contours = cf.find_contours(img_thresh, square=True)

        for cnt, pts in contours:
            pts = mf.arrange_points(pts)
            frame = cf.warp_image(img_thresh, pts, frame)
            frame_inv = cf.invert(frame)
            frame_inv = cf.crop_square(frame_inv, 20)

            shapes = len(cf.find_contours(frame_inv, 0, 0))
            place = ""
            if shapes == 2:
                place = "Pasillo"
            elif shapes == 3:
                place = "Sala"
            else:
                place = "Cocina"
            printer.print_poly(img, pts, thickness=8)
            printer.print_text(img,
                               pts[0][0] - 40,
                               40,
                               ("Frame detected", (255, 0, 0), 3),
                               (f"{shapes} detected", (0, 255, 0), 3),
                               (f"{place}", (0,0,255), 3))


        cv2.imshow("a", frame)
        cv2.imshow("b", img)
        cv2.imshow("c", img_thresh)

