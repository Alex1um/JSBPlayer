import cv2
from src.detect_danger import detect_danger
from src.preprocess import preprocess
from src.detect_enemies import detect_enemies
from src.detect_player import detect_player
import numpy as np
from src.dash import get_dash_coords, DASH_DISTANCE
from src.action import get_action


prev_player = None


def get_policy(frame, draw_enemies=False, crop_black=False) -> tuple[list[list[int,int,int]], list[int | None, int | None], bool]:
    global prev_player
    frame = preprocess(frame, crop_black=crop_black)
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    player = detect_player(hsv_frame)
    if player is None:
        player = prev_player
    if player is None:
        return frame, (None, None), False
    prev_player = player
    xp, yp = player
    enemies, radiuses, rects, rect_radiuses, enemy_contours = detect_enemies(hsv_frame, player)
    dangerous_countours, dangerous_rects, dangerous_radiuses = detect_danger(hsv_frame, player)

    all_enemies = np.array(enemies + rects + dangerous_rects)
    all_radiuses = np.array(radiuses + rect_radiuses + dangerous_radiuses)
    
    if draw_enemies:
        for enemy, radius in zip(all_enemies, all_radiuses):
            cv2.circle(frame, (enemy[0], enemy[1]), int(radius) if radius > 0 else 5, (0, 255, 0), -1)

    if player is not None:
        cv2.circle(frame, (xp, yp), 10, (0, 0, 255), -1)

    cv2.drawContours(frame, dangerous_countours, -1, (0, 255, 255), 2)
    cv2.drawContours(frame, enemy_contours, -1, (255, 0, 255), 2)

    is_dash = False
    new_x, new_y = None, None
    if len(enemies) > 0:
        dash_coords = get_dash_coords(frame, player, dangerous_countours, enemy_contours, all_enemies, all_radiuses)
        if dash_coords is not None:
            new_x, new_y = dash_coords
            is_dash = True
            cv2.circle(frame, (xp + new_x * 30, yp + new_y * 30), 10, (255, 255, 0), -1)
        else:
            new_x, new_y = get_action(player, all_enemies, all_radiuses, center)
            cv2.circle(frame, (xp + new_x * 10, yp + new_y * 10), 10, (255, 255, 0), -1)
    return frame, (new_x, new_y), is_dash