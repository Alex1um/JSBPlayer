import cv2
from math import pi
from collections import namedtuple

_lower_background_hsv = (160, 0, 200)
_upper_background_hsv = (180, 255, 255)
rectEnemy = namedtuple('rectEnemy', ['x', 'y', 'w', 'h'])

def detect_enemies(hsv_frame) -> list[tuple[int, int]]:
    fh, fw, *_ = hsv_frame.shape
    mask = cv2.inRange(hsv_frame, _lower_background_hsv, _upper_background_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the coordinates of the centroid of the biggest contour
    objects = []
    rectangular = []
    radiuses = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            radius = 0
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h
            aspect2 = h / w
            circularity = 0
            if h > 0 and w > 0 and aspect > 20 or aspect2 > 20 or w > fw * 0.9 or h > fh * 0.9: # rects
                rectangular.append(rectEnemy(x, y, w, h))
                continue
            elif perimeter > 0: # circles
                area = cv2.contourArea(contour)
                circularity = 4 * pi * area / (perimeter * perimeter)
                if circularity > 0.5 and perimeter > fh * 0.1:
                    radius = perimeter / (2 * pi)
            radiuses.append(radius)
            objects.append((center_x, center_y))
    
    return objects, rectangular, radiuses, contours
