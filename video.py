import cv2
from src.policy import get_policy

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: ({x}, {y})")

def start():
    record = True
    # cap = cv2.VideoCapture("./chronos-big.mp4")
    # cap = cv2.VideoCapture("./light.mp4")
    # cap = cv2.VideoCapture("./dark.mp4")
    # cap = cv2.VideoCapture("./darker.mp4")
    # cap = cv2.VideoCapture("./darkest.mp4")
    cap = cv2.VideoCapture("./presd/detectraw.mp4")
    cv2.namedWindow('frame')
    cv2.setMouseCallback("frame", mouse_callback)
    if record:
        out = cv2.VideoWriter('filename.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         20, (960, 480))
    else:
        out = None
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame, action, is_dash = get_policy(
            frame,
            crop_black=True,
            debug_danger=False,
            debug_dash=False,
            debug_move=False,
            debug_enemies=False,
            debug_tracks=False,
            draw_enemies=False,
            use_tracking=False,
            debug_tracking_points=False,
            debug_simple_enemies=False,
            draw_player=False,
        )
        if record:
            out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(100000) == ord("q"):
            break
    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start()