import cv2
import cv2
from src.policy import get_policy
import numpy as np
import keyboard

PRESS = True

x_key = {
    1: "d",
    -1: "a",
}

y_key = {
    1: "s",
    -1: "w",
}

def start_press():
    global PRESS
    print(1)
    PRESS = not PRESS

def start():
    global PRESS
    cap = cv2.VideoCapture("/dev/video0")
    # cap = cv2.VideoCapture("./no_sound.mp4")
    current_key_x, current_key_y = 0, 0
    while True:
        ret, frame = cap.read()
        frame, action, is_dash = get_policy(
            frame,
            crop_black=False,
            debug_danger=True,
            debug_dash=False,
            debug_move=False,
            debug_enemies=True,
            debug_tracks=True,
            draw_enemies=True,
            draw_player=True,
        )
        new_x, new_y = action
        if PRESS:
            if new_x is not None and new_y is not None:
                if current_key_x != new_x:
                    if (old_key := x_key.get(current_key_x, None)) is not None:
                        keyboard.release(old_key)
                    if (new_key := x_key.get(new_x, None)) is not None:
                        keyboard.press(new_key)
                    current_key_x = new_x
                if current_key_y != new_y:
                    if (old_key := y_key.get(current_key_y, None)) is not None:
                        keyboard.release(old_key)
                    if (new_key := y_key.get(new_y, None)) is not None:
                        keyboard.press(new_key)
                    current_key_y = new_y
            if is_dash:
                keyboard.press_and_release("space")
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("r"):
            start_press()


if __name__ == "__main__":
    # start()
    keyboard.add_hotkey("enter", start)
    keyboard.add_hotkey("r", start_press)
    keyboard.wait("q")