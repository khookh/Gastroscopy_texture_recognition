import sys
import time
import cv2 as cv
import numpy as np
from threading import Thread
from multiprocessing import Queue


def crop(img):
    size = img.shape
    h, s, v = cv.split(img)
    x1 = 1
    x2 = size[1] - 1
    y1 = 1
    y2 = size[0] - 1
    while v[int(size[0] / 2), x1] < 15:
        x1 += 5
    while v[int(size[0] / 2), x2] < 15:
        x2 -= 5
    while v[y1, int(size[1] / 2)] < 15:
        y1 += 5
    while v[y2, int(size[1] / 2)] < 15:
        y2 -= 5
    return [y1, y2, x1, x2]


def read_flux():
    """
    Thread fetching frames
    """
    global count, cap
    c_s = []
    fps_list = np.array([])
    ratio = 1
    while not cap.isOpened():  # active waiting if the input isn't ready yet (for real-time)
        if str(sys.argv[2]) == "rpi":
            cap = cv.VideoCapture(0)
        else:
            cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(500)
    while True:
        start = time.time()
        cv.waitKey(1)
        ret, frame = cap.read()
        if count == 1:
            c_s = crop(frame)
            ratio = 216 / frame.shape[0]
        if ret:
            frame = cv.resize(frame[c_s[0]:c_s[1], c_s[2]:c_s[3]], None, fx=ratio, fy=ratio,
                              interpolation=cv.INTER_CUBIC)
            q_frame.put(frame)
            cv.imshow('window', frame)
            count += 1
            end = time.time()
            fps_list = np.append(fps_list, end - start)
        else:
            break
    cap.release()
    print(np.mean(fps_list))
    # result laptop = 62.34


if __name__ == '__main__':
    count = 1
    q_frame = Queue()
    if str(sys.argv[2]) == "rpi":  # temporaire
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(str(sys.argv[1]))
    thread_fetch = Thread(target=read_flux)
    thread_fetch.start()
    thread_fetch.join()
