import cv2

_lower_background_hsv = (160, 0, 50)
_upper_background_hsv = (180, 255, 255)

def detect_enemies(hsv_frame) -> list[tuple[int, int]]:
    mask = cv2.inRange(hsv_frame, _lower_background_hsv, _upper_background_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the coordinates of the centroid of the biggest contour
    objects = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            objects.append((center_x, center_y))
    
    return objects, contours
