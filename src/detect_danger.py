import cv2

_lower_background_hsv = (120, 40, 65)
_upper_background_hsv = (230, 255, 200)


def detect_danger(hsv_frame):
    fh, fw, *_ = hsv_frame.shape
    mask = cv2.inRange(hsv_frame, _lower_background_hsv, _upper_background_hsv)
    # cv2.imshow("danger mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = list()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > fw * 0.9 or h > fh * 0.9: # rects
            rects.append((x, y, w, h))

    return contours, rects
