import cv2
from src.detect_player import detect_player
from src.detect_enemies import detect_enemies
from src.detect_danger import detect_danger
from src.dash import get_dash_coords
from src.action import get_action
from src.preprocess import preprocess
import numpy as np
import keyboard


x_key = {
    1: "d",
    -1: "a",
}

y_key = {
    1: "s",
    -1: "w",
}

def start():
    cap = cv2.VideoCapture("/dev/video0")
    # cap = cv2.VideoCapture("./no_sound.mp4")
    current_key_x, current_key_y = 0, 0
    while True:
        ret, frame = cap.read()
        frame = preprocess(frame)
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        player, player_countours = detect_player(hsv_frame)
        if player is None:
            continue
        xp, yp = player
        enemies, rects, radiuses, contours = detect_enemies(hsv_frame)
        dangerous_countours = detect_danger(hsv_frame)
        for x, y, w, h in rects: # rects
            if h > w:
                on_side_points = (x if x > xp else x + w, yp)
            else:
                on_side_points = (xp, y if y > yp else y + h)
            enemies.append(on_side_points)
            radiuses.append(0)
        enemies, radiuses = np.array(enemies), np.array(radiuses)

        for enemy in enemies:
            cv2.circle(frame, (enemy[0], enemy[1]), 10, (0, 255, 0), -1)
        
        new_x, new_y = None, None
        if len(enemies) > 0:
            dash_coords = get_dash_coords((w, h), player, dangerous_countours, enemies, radiuses, rects)
            if dash_coords is not None:
                new_x, new_y = dash_coords
                cv2.circle(frame, (xp + new_x * 20, yp + new_y * 20), 10, (255, 255, 0), -1)
            else:
                new_x, new_y = get_action(player, enemies, rects, radiuses, center)
                cv2.circle(frame, (xp + new_x * 10, yp + new_y * 10), 10, (0, 255, 0), -1)
        if new_x is not None and new_y is not None:
            if current_key_x != new_x:
                keyboard.release(x_key[current_key_x])
                if (new_key := x_key.get(new_x, None)) is not None:
                    keyboard.press(new_key)
                current_key_x = new_x
            if current_key_y != new_y:
                keyboard.release(y_key[current_key_y])
                if (new_key := y_key.get(new_y, None)) is not None:
                    keyboard.press(new_key)
                current_key_y = new_y
        cv2.imshow("frame", frame)
        if cv2.waitKey(1000) == ord("q"):
            break


if __name__ == "__main__":
    # start()
    keyboard.add_hotkey("enter", start)
    keyboard.wait("q")