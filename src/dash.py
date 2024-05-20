import cv2
import numpy as np

DASH_DISTANCE = 100
SEARCH_RADIUS = 500


def is_dash_needed(player_pos, dangerous_countours, enemies, radiuses):
    for countour in dangerous_countours:
        if cv2.pointPolygonTest(countour, player_pos, False) >= 0:
            return True
    ranges = np.linalg.norm(np.array(enemies) - np.array(player_pos), axis=1)
    indexes = ranges < SEARCH_RADIUS
    ranges = ranges[indexes] - radiuses[indexes]
    return np.any(ranges <= 0)


def check_if_point_safe(frame_size, point, dangerous_countours, enemy_contours, objects, radiuses):
    if (
        point[0] < 0
        or point[0] >= frame_size[0]
        or point[1] < 0
        or point[1] >= frame_size[1]
    ):
        return -float("inf")
    s = 0
    for countour in dangerous_countours:
        if cv2.pointPolygonTest(countour, point, False) >= 0:
            s = -200
    for countour in enemy_contours:
        if cv2.pointPolygonTest(countour, point, False) >= 0:
            s -= 1000

    # for x, y, w, h in enemy_rects:
    #     if x <= point[0] <= x + w and y <= point[1] <= y + h:
    #         return False
    ranges = np.linalg.norm(np.array(objects) - np.array(point), axis=1)
    indexes = ranges < SEARCH_RADIUS
    ranges = ranges[indexes] - radiuses[indexes]
    return np.sum(ranges) + s


def get_dash_coords(frame_size, player_pos, dangerous_countours, enemy_contours, objects, radiuses):
    if not is_dash_needed(player_pos, dangerous_countours, objects, radiuses):
        return None
    available_points = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    # real_points = available_points * DASH_DISTANCE + player_pos
    return max(
        available_points,
        key=lambda x: check_if_point_safe(
            frame_size,
            (
                player_pos[0] + x[0] * DASH_DISTANCE,
                player_pos[1] + x[1] * DASH_DISTANCE,
            ),
            dangerous_countours,
            enemy_contours,
            objects,
            radiuses,
        ),
    )
    # for available_point in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
    #     point = (player_pos[0] + available_point[0] * DASH_DISTANCE, player_pos[1] + available_point[1] * DASH_DISTANCE)
    # if check_if_point_safe(frame_size, point, dangerous_countours, objects, radiuses):
    #     return available_point
    return None
