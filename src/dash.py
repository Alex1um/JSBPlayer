import cv2
import numpy as np

DASH_DISTANCE = 100
SEARCH_RADIUS = 1000


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
        return -99999999
    s = 0
    for countour in dangerous_countours:
        if cv2.pointPolygonTest(countour, point, False) >= 0:
            s = -5000
    for countour in enemy_contours:
        if cv2.pointPolygonTest(countour, point, False) >= 0:
            s -= 20000

    # for x, y, w, h in enemy_rects:
    #     if x <= point[0] <= x + w and y <= point[1] <= y + h:
    #         return False
    ranges = np.linalg.norm(np.array(objects) - np.array(point), axis=1)
    indexes = ranges < SEARCH_RADIUS
    ranges = ranges[indexes] - radiuses[indexes]
    rs = np.sum(ranges) + s
    return rs if rs != 0 else 50000


def get_dash_coords(frame, player_pos, dangerous_countours, enemy_contours, objects, radiuses):
    if not is_dash_needed(player_pos, dangerous_countours, objects, radiuses):
        return None
    h, w, _ = frame.shape
    frame_size = (w, h)
    available_points = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    mx = 0
    mx_p = None
    for point in available_points:
        real_point = (player_pos[0] + point[0] * DASH_DISTANCE, player_pos[1] + point[1] * DASH_DISTANCE)
        is_save = check_if_point_safe(frame_size, real_point, dangerous_countours, enemy_contours, objects, radiuses)
        cv2.circle(frame, real_point, 10, (0, 0, 255), -1)
        cv2.addText(frame, f"{is_save:.2f} {real_point}", real_point, "Arial", 10, (255, 255, 255))
        if is_save > mx:
            mx = is_save
            mx_p = point
    return mx_p
    # return max(
    #     available_points,
    #     key=lambda x: check_if_point_safe(
    #         frame_size,
    #         (
    #             player_pos[0] + x[0] * DASH_DISTANCE,
    #             player_pos[1] + x[1] * DASH_DISTANCE,
    #         ),
    #         dangerous_countours,
    #         enemy_contours,
    #         objects,
    #         radiuses,
    #     ),
    # )

    # for available_point in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
    #     point = (player_pos[0] + available_point[0] * DASH_DISTANCE, player_pos[1] + available_point[1] * DASH_DISTANCE)
    # if check_if_point_safe(frame_size, point, dangerous_countours, objects, radiuses):
    #     return available_point
    return None
