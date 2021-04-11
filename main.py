import cv2
import numpy as np


def nothing(x):
    pass



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
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, imgThresh = cv2.threshold(imgGray, 217, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, perimeter * 0.02, True)
                if len(approx) == 4:  # Need to find the aspect ratio
                    x = approx[0][0][0]
                    y = approx[2][0][1]
                    aspectRatio = y / x
                    if aspectRatio > 0.8:

                        points = arrangePoints(approx)
                        cv2.polylines(img, points, True, (0, 255, 0), 7, cv2.LINE_AA)
                        cv2.putText(img, "Detected", (points[0][0][0], points[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 3)


        cv2.imshow("Thresh", imgThresh)
        cv2.imshow("c", img)

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
