import cv2
import numpy as np


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def findContours(src, dst, iterations=3, kernel=(100, 100), minArea=7000, polyFilter=None, showContours=False):

    # Preprocess the image

    imgGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, kernel[0], kernel[1])
    imgDilat = cv2.dilate(imgCanny, np.ones((5,5)), iterations=iterations)
    imgErode = cv2.erode(imgDilat, np.ones((5,5)), iterations=iterations-1)

    cv2.imshow("Canny", imgErode)
    # Find contours

    filteredContours = []
    contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cntArea = cv2.contourArea(cnt)
        if cntArea > minArea:
            cntPerimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, cntPerimeter * 0.02, True)
            if polyFilter is None:
                filteredContours.append((cntArea, cntPerimeter, approx, len(approx), cnt))
            elif len(approx) == polyFilter:
                filteredContours.append((cntArea, cntPerimeter, approx, len(approx), cnt))

    # Display Contours

    if showContours:
        for cnt in filteredContours:
            cv2.drawContours(dst, cnt[4], -1, (0, 255, 0), 10)

    cv2.imshow("Contours", dst)
    return filteredContours












