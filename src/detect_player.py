import cv2

_lower_background_hsv = (70, 220, 220)
_upper_background_hsv = (100, 255, 255)

def detect_player(hsv_frame) -> tuple[tuple[int, int] | None, list]:
    mask = cv2.inRange(hsv_frame, _lower_background_hsv, _upper_background_hsv)
    cv2.rectangle(mask, (275, 5), (685, 25), (0, 0, 0), cv2.FILLED)
    cv2.imshow("mask", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the coordinates of the centroid of the biggest contour
    if len(contours) > 0:
        # contours = max(contours, key=cv2.contourArea)
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return (center_x, center_y)
    
    return (None)
