import cv2


def preprocess(frame):
    # crop upper 10%
    # h, w, _ = frame.shape
    # frame = frame[int(h * 0.1) : h, :, :]
    frame = cv2.resize(frame, (960, 540))
    return frame