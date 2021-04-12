import numpy as np
import math


def calculate_distance(pt1, pt2):
    return math.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)


def arrange_points(pts):

    points_rzd = pts.copy()
    points_rzd.resize(4, 2)
    newPts = np.zeros((4, 1, 2), np.int32)

    add = points_rzd.sum(axis=1)
    newPts[0] = points_rzd[np.argmin(add)]
    newPts[3] = points_rzd[np.argmax(add)]

    diff = np.diff(points_rzd, axis=1)
    newPts[1] = points_rzd[np.argmin(diff, axis=0)]
    newPts[2] = points_rzd[np.argmax(diff, axis=0)]

    return newPts