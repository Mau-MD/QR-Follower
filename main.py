import cv2
import numpy as np
import math


def nothing(x):
    pass


def calculateDistance(pt1, pt2):
    return math.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)


def arrangePoints(pts):
    pts.resize(4, 2)
    newPts = np.zeros((4, 1, 2), np.int32)

    add = pts.sum(axis=1)
    newPts[0] = pts[np.argmin(add)]
    newPts[3] = pts[np.argmax(add)]

    diff = np.diff(pts, axis=1)
    newPts[1] = pts[np.argmin(diff, axis=0)]
    newPts[2] = pts[np.argmax(diff, axis=0)]

    return newPts


def main():
    cap = cv2.VideoCapture(0)

    # Read Resources
    frameSize = 250
    resizeFactor = 16
    imgSala = cv2.imread("resources/Sala.png")
    imgSala = cv2.resize(imgSala, (imgSala.shape[1] // resizeFactor, imgSala.shape[0] // resizeFactor))
    imgSala = imgSala[10:2500 // resizeFactor - 5, 10:imgSala.shape[1] - 10]

    # cv2.namedWindow("Trackbars",)
    # cv2.createTrackbar("lh","Trackbars",0,179,nothing)
    # cv2.createTrackbar("ls","Trackbars",0,255,nothing)
    # cv2.createTrackbar("lv","Trackbars",0,255,nothing)
    # cv2.createTrackbar("uh","Trackbars",179,179,nothing)
    # cv2.createTrackbar("us","Trackbars",255,255,nothing)
    # cv2.createTrackbar("uv","Trackbars",255,255,nothing)

    while True:

        _, img = cap.read()
        # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, imgThresh = cv2.threshold(imgGray, 217, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        whiteFrame = np.zeros((frameSize, frameSize))
        drawFrame = np.zeros_like(whiteFrame)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, perimeter * 0.02, True)
                if len(approx) == 4:  # Need to find the aspect ratio

                    points = arrangePoints(approx)
                    # print(points[0][0], points[1][0], points[2][0])
                    w, h = calculateDistance(points[0][0], points[1][0]), calculateDistance(points[0][0], points[2][0])
                    # print(w, h)
                    if w == 0 or h == 0: continue
                    aspectRatio = w / h
                    if aspectRatio > 0.90:
                        src = np.float32(points)
                        dst = np.float32([[0, 0], [frameSize, 0], [0, frameSize], [frameSize, frameSize]])
                        transform = cv2.getPerspectiveTransform(src, dst)
                        whiteFrame = cv2.warpPerspective(imgThresh, transform, (frameSize, frameSize))

                        # Find Contours again
                        imgInvThresh = (255 - whiteFrame)
                        imgInvThresh = imgInvThresh[20:imgInvThresh.shape[1] - 20, 20 : imgInvThresh.shape[0] - 20]
                        figureContours, _ = cv2.findContours(imgInvThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(drawFrame, figureContours, -1, (255, 0, 0), 5)
                        # print(len(figureContours))


                        # Drawing
                        figures = len(figureContours)
                        cv2.polylines(img, points, True, (0, 255, 0), 7, cv2.LINE_AA)
                        cv2.putText(img, f"Frame Detected. {figures} Figures Detected",
                                    (points[0][0][0], points[0][0][1] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 119, 15), 3)
                        cv2.putText(img, f"Area: {area}",
                                    (points[0][0][0], points[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 200, 15), 3)

                        if figures == 2:
                            cv2.putText(img, "Pasillo",
                                        (points[0][0][0], points[0][0][1] - 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 3)
                        if figures == 3:
                            cv2.putText(img, "Sala",
                                        (points[0][0][0], points[0][0][1] - 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 3)
                        if figures == 4:
                            cv2.putText(img, "Cocina",
                                        (points[0][0][0], points[0][0][1] - 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 3)



        cv2.imshow("Threshold", imgThresh)
        cv2.imshow("Camera", img)
        cv2.imshow("White Frame", whiteFrame)
        cv2.imshow("Draw Frame", drawFrame)

        # Nopeeee
        # mask = (255 - imgSala)
        # match = cv2.matchTemplate(img, imgSala, cv2.TM_SQDIFF, mask=mask)
        # # cv2.normalize(match, match, 0 , 1, cv2.NORM_MINMAX, -1)
        #
        # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(match)
        # matchLoc = minLoc
        #
        # cv2.rectangle(img, matchLoc, (matchLoc[0] + 20, matchLoc[1] + 20), (255, 0 , 0), 3)
        #
        # cv2.imshow("img", img)

        # Nope
        # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #
        # lh = cv2.getTrackbarPos("lh","Trackbars")
        # ls = cv2.getTrackbarPos("ls","Trackbars")
        # lv = cv2.getTrackbarPos("lv","Trackbars")
        # uh = cv2.getTrackbarPos("uh","Trackbars")
        # us = cv2.getTrackbarPos("us","Trackbars")
        # uv = cv2.getTrackbarPos("uv","Trackbars")
        #
        # lower_gray = np.array([lh, ls, lv], np.uint8)
        # upper_gray = np.array([uh, us, uv], np.uint8)
        #
        # mask_gray = cv2.inRange(imgHSV, lower_gray, upper_gray)
        # mask_gray = (255 - mask_gray)
        # cv2.imshow("gray", mask_gray)

        # Didn't work either
        # res = cv2.matchTemplate(img, imgSala, cv2.TM_CCOEFF)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #
        # top_left = max_loc
        # bottom_right = (top_left[0] + 100, top_left[1] + 100)
        #
        # cv2.rectangle(img, top_left, bottom_right, 255, 2)
        # cv2.imshow("Template", img)

        # Didn't work
        # orb = cv2.ORB_create(nfeatures=10)
        # kp1, des1 = orb.detectAndCompute(img, None)
        # kp2, des2 = orb.detectAndCompute(imgSala, None)
        #
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        #
        # match_img = cv2.drawMatches(img, kp1, imgSala, kp2, matches[:50], None)
        #
        # cv2.imshow("Matches", match_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
