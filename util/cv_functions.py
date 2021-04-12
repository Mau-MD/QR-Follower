import cv2
import numpy as np
from . import math_functions as mf


def show_images(frames, size=1):
    i = 1
    for frame in frames:
        frame = cv2.resize(frame, int(frame.shape[1] * size), int(frame.shape[0] * size))
        cv2.imshow(f"Image {i}", frame)
        i += 1


def is_a_square(pts):
    points = mf.arrange_points(pts)
    w, h = mf.calculate_distance(points[0][0], points[1][0]), mf.calculate_distance(points[0][0], points[2][0])
    if w == 0 or h == 0:
        return False
    aspect_ratio = w / h
    if aspect_ratio > 0.90:
        return True


def warp_image(img, src, frame):
    w, h = frame.shape[1], frame.shape[0]
    src = np.float32(src)
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    transform = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, transform, (w, h))
    return img


def crop_square(img, px):
    img = img[px:img.shape[1] - px, px:img.shape[0] - px]
    return img


def invert(img):
    return 255 - img


def find_contours(img, max_area=1000, poly=4, square=False):
    contours_list = []
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > max_area:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, perimeter * 0.02, True)

            if poly != 0:
                if len(approx) == poly:
                    if not square or (square and is_a_square(approx)):
                        contours_list.append((cnt, approx))
            else:
                contours_list.append((cnt, approx))

    return contours_list




