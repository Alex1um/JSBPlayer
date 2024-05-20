import cv2

# _lower_background_hsv = (120, 40, 65)
_lower_background_hsv = (150, 40, 40)
_upper_background_hsv = (230, 255, 200)


def detect_danger(hsv_frame, player_pos):
    fh, fw, *_ = hsv_frame.shape
    xp, yp = player_pos
    mask = cv2.inRange(hsv_frame, _lower_background_hsv, _upper_background_hsv)
    # cv2.imshow("danger mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = list()
    rects_radiuses = list()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > fw * 0.9 or h > fh * 0.9: # rects
            if h > w:
                on_side_points = (x if x > xp else x + w, yp)
            else:
                on_side_points = (xp, y if y > yp else y + h)
            rects.append(on_side_points)
            rects_radiuses.append(10)

    return contours, rects, rects_radiuses
