import cv2


# (Text, Color, Size)
def print_text(img, pt, offset=40, *args):
    i = 0
    for text in args:
        cv2.putText(img, text[0], (pt[0], pt[1] + i * offset), cv2.FONT_HERSHEY_SIMPLEX, 1, text[1], text[2])
        i += 1


def print_poly(img, pts, color=(255, 0, 0), thickness=3):
    cv2.polylines(img, pts, True, color, thickness)
