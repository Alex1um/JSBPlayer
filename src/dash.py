import cv2
import numpy as np

DASH_DISTANCE = 100

def is_dash_needed(player_pos, dangerous_countours):
    for countour in dangerous_countours:
        if cv2.pointPolygonTest(countour, player_pos, False) >= 0:
            return True
    return False


def check_if_point_safe(frame_size, point, dangerous_countours, enemy_rects, radiuses, objects):
    if point[0] < 0 or point[0] >= frame_size[0] or point[1] < 0 or point[1] >= frame_size[1]:
        return False
    for countour in dangerous_countours:
        if cv2.pointPolygonTest(countour, point, False) >= 0:
            return False
    for x, y, w, h in enemy_rects:
        if x <= point[0] <= x + w and y <= point[1] <= y + h:
            return False
    return not np.any(np.linalg.norm(np.array(objects) - np.array(point), axis=1) < radiuses)


def get_dash_coords(frame_size, player_pos, dangerous_countours, enemy_rects, radiuses, objects):
    if not is_dash_needed(player_pos, dangerous_countours):
        return None
    for available_point in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        point = (player_pos[0] + available_point[0] * DASH_DISTANCE, player_pos[1] + available_point[1] * DASH_DISTANCE)
        if check_if_point_safe(frame_size, point, dangerous_countours, enemy_rects, radiuses, objects):
            return available_point
    return None


