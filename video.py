import cv2
from src.detect_player import detect_player
from src.detect_enemies import detect_enemies
from src.detect_danger import detect_danger
from src.dash import get_dash_coords
from src.action import get_action
from src.preprocess import preprocess
import numpy as np


def start():
    # cap = cv2.VideoCapture("./chronos-big.mp4")
    # cap = cv2.VideoCapture("./light.mp4")
    # cap = cv2.VideoCapture("./dark.mp4")
    # cap = cv2.VideoCapture("./darker.mp4")
    cap = cv2.VideoCapture("./darkest.mp4")
    prev_player = None
    while True:
        ret, frame = cap.read()
        frame = preprocess(frame)
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        player = detect_player(hsv_frame)
        if player is None:
            player = prev_player
        if player is None:
            continue
        prev_player = player
        xp, yp = player
        enemies, radiuses, rects, rect_radiuses, enemy_contours = detect_enemies(hsv_frame, player)
        dangerous_countours, dangerous_rects, dangerous_radiuses = detect_danger(hsv_frame, player)

        all_enemies = np.array(enemies + rects + dangerous_rects)
        all_radiuses = np.array(radiuses + rect_radiuses + dangerous_radiuses)
        
        for enemy, radius in zip(all_enemies, all_radiuses):
            cv2.circle(frame, (enemy[0], enemy[1]), int(radius) if radius > 0 else 5, (0, 255, 0), -1)

        if player is not None:
            cv2.circle(frame, (xp, yp), 10, (0, 0, 255), -1)

        cv2.drawContours(frame, dangerous_countours, -1, (0, 255, 255), 2)
        cv2.drawContours(frame, enemy_contours, -1, (255, 0, 255), 2)
        
        if len(enemies) > 0:
            dash_coords = get_dash_coords((w, h), player, dangerous_countours, enemy_contours, all_enemies, all_radiuses)
            if dash_coords is not None:
                new_x, new_y = dash_coords
                cv2.circle(frame, (xp + new_x * 20, yp + new_y * 20), 10, (255, 255, 0), -1)
            else:
                new_x, new_y = get_action(player, all_enemies, all_radiuses, center)
                cv2.circle(frame, (xp + new_x * 10, yp + new_y * 10), 10, (255, 255, 0), -1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1000) == ord("q"):
            break


if __name__ == "__main__":
    start()