import cv2
from src.detect_player import detect_player
from src.detect_enemies import detect_enemies
from src.action import get_action
from src.preprocess import preprocess
import numpy as np


def start():
    # cap = cv2.VideoCapture("/dev/video0")
    cap = cv2.VideoCapture("./countours.mp4")
    while True:
        ret, frame = cap.read()
        frame = preprocess(frame)
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        player, player_countours = detect_player(hsv_frame)
        enemies, rects, radiuses, contours = detect_enemies(hsv_frame)
        # cv2.drawContours(frame, player_countours, -1, (0, 255, 0), 3)
        if player is not None:
            xp, yp = player
            for x, y, w, h in rects:
                if h > w:
                    on_side_points = (x if x > xp else x + w, yp)
                else:
                    on_side_points = (xp, y if y > yp else y + h)
                enemies.append(on_side_points)
                radiuses.append(0)

        for enemy in enemies:
            cv2.circle(frame, (enemy[0], enemy[1]), 10, (0, 255, 0), -1)
        if player is not None and len(enemies) > 0:
            x, y = player
            new_x, new_y = get_action(player, enemies, rects, radiuses, center)
            cv2.circle(frame, (x + new_x * 10, y + new_y * 10), 10, (0, 255, 0), -1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1000) == ord("q"):
            break


if __name__ == "__main__":
    start()