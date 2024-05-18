import cv2

_lower_background_hsv = (160, 20, 50)
_upper_background_hsv = (180, 200, 200)


def detect_danger(hsv_frame):
    mask = cv2.inRange(hsv_frame, _lower_background_hsv, _upper_background_hsv)
    cv2.imshow("danger mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
