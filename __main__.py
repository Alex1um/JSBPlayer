import cv2
from src.detect_player import detect_player
from src.detect_enemies import detect_enemies
from src.action import get_action
import keyboard


def preprocess(frame):
    # crop upper 10%
    h, w, _ = frame.shape
    frame = frame[int(h * 0.1) : h, :, :]
    frame = cv2.resize(frame, (640, 480))
    return frame


def start():
    # cap = cv2.VideoCapture("/dev/video0")
    cap = cv2.VideoCapture("./no_sound.mp4")
    while True:
        ret, frame = cap.read()
        frame = preprocess(frame)
        h, w, _ = frame.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        player, player_countours = detect_player(hsv_frame)
        enemies, contours = detect_enemies(hsv_frame)
        cv2.drawContours(frame, player_countours, -1, (0, 255, 0), 3)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
        if player is not None:
            x, y = player
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        if player is not None and len(enemies) > 0:
            new_x, new_y = get_action(player, enemies)
            cv2.circle(frame, (x + new_x * 10, y + new_y * 10), 10, (0, 255, 0), -1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1000) == ord("q"):
            break


if __name__ == "__main__":
    # start()
    keyboard.add_hotkey("enter", start)
    keyboard.wait("q")