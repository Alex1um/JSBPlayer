import cv2
from src.policy import get_policy


def start():
    # cap = cv2.VideoCapture("./chronos-big.mp4")
    # cap = cv2.VideoCapture("./light.mp4")
    cap = cv2.VideoCapture("./dark.mp4")
    # cap = cv2.VideoCapture("./darker.mp4")
    # cap = cv2.VideoCapture("./darkest.mp4")
    while True:
        ret, frame = cap.read()
        frame, action, is_dash = get_policy(
            frame,
            crop_black=True,
            debug_danger=True,
            debug_dash=True,
            debug_move=True,
            debug_enemies=True,
            debug_tracks=True,
            draw_enemies=True,
            use_tracking=True,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(100000) == ord("q"):
            break


if __name__ == "__main__":
    start()