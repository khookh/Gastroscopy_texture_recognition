import sys
import time
import cv2 as cv
import numpy as np
from threading import Thread


def treatment(img):  # some library call, no logic here
    kernel = np.ones((7, 7), np.uint8)
    img = cv.GaussianBlur(img, (5, 5), 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img, (0, 35, 170), (60, 100, 250))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # clustering
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)  # de-noise
    return cv.findNonZero(mask)


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
    tps_list = np.array([])
    ratio = 1
    while not cap.isOpened():  # active waiting if the input isn't ready yet (for real-time)
        if str(sys.argv[2]) == "rpi":
            cap = cv.VideoCapture(0)
        else:
            cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(500)
    while True:
        cv.waitKey(1)
        ret, frame = cap.read()
        if count == 1:
            c_s = crop(frame)
            ratio = 216 / frame.shape[0]
        if ret:
            frame = cv.resize(frame[c_s[0]:c_s[1], c_s[2]:c_s[3]], None, fx=ratio, fy=ratio,
                              interpolation=cv.INTER_CUBIC)
            start = time.time()
            treatment(frame)
            end = time.time()
            tps = 1 / (end - start)
            tps_list = np.append(tps_list, tps)
            count += 1
        else:
            break
    cap.release()
    print(np.mean(tps_list))


if __name__ == '__main__':
    count = 1
    if str(sys.argv[2]) == "rpi":
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(str(sys.argv[1]))
    thread_fetch = Thread(target=read_flux)
    thread_fetch.start()
    thread_fetch.join()
