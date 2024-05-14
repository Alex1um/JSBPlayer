import cv2
from src.detect_player import detect_player
from src.detect_enemies import detect_enemies
from src.action import get_action
from src.preprocess import preprocess
import numpy as np
from ewmh import EWMH
from Xlib import display, X
from Xlib.protocol.request import GetImage


def capture_window(disp, window):
    geom = window.get_geometry()
    x, y, width, height = geom.x, geom.y, geom.width, geom.height
    print(x, y, width, height)
    # screenshot = disp.grab_area(x, y, width, height)
    # screenshot = disp.capture_screen(x, y, width, height)
    # image = cv2.cvtColor(np.array(screenshot.get_image(), dtype=np.uint8), cv2.COLOR_BGR2RGB)
    # return image

if __name__ == "__main__":
    disp = display.Display()
    screen = disp.screen()
    root = screen.root
    children = root.query_tree().children
    ewmh = EWMH(disp, root)
    for win in ewmh.getClientList():
        pid = ewmh.getWmPid(win)
        print(pid, win.get_wm_name())
        geom = win.get_geometry()
        x, y, width, height = geom.x, geom.y, geom.width, geom.height
        img = win.get_image(x, y, width, height, X.ZPixmap, 0)
        img = str(img)
        print(img.encode()[:100])
    # while True:
    #     frame = capture_window("Firefox Web Browser")
    #     frame = preprocess(frame)
    #     h, w, _ = frame.shape
    #     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     player, player_countours = detect_player(hsv_frame)
    #     enemies, contours = detect_enemies(hsv_frame)
    #     cv2.drawContours(frame, player_countours, -1, (0, 255, 0), 3)
    #     cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    #     if player is not None:
    #         x, y = player
    #         cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    #     if player is not None and len(enemies) > 0:
    #         new_x, new_y = get_action(player, enemies)
    #         cv2.circle(frame, (x + new_x * 10, y + new_y * 10), 10, (0, 255, 0), -1)
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1000) == ord("q"):
    #         break
