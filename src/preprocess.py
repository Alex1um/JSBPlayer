import cv2
import numpy as np


def preprocess(frame, crop_black = False):
    # crop upper 10%
    # h, w, _ = frame.shape
    # frame = frame[int(h * 0.1) : h, :, :]
    frame = cv2.resize(frame, (960, 540))
    if crop_black:
        x, y, w, h = 0, 30, 960, 480
        return frame[y : y + h, x : x + w]
    return frame
