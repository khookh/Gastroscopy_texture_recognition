from multiprocessing import Process, Queue
import cv2 as cv
import numpy as np


# segmentation (HSV)
def seg_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # meth.display_hist(img,count)
    h, s, v = cv.split(img)
    # temp seg masks
    mask = cv.inRange(img, (0, 35, 170), (60, 100, 245))  # direct light
    mask2 = cv.inRange(img, (0, 0, 90), (30, 95, 170))  # low light foam
    return mask + mask2
    # return cv.bitwise_and(img, img, mask=mask)


kernel = np.ones((7, 7), np.uint8)
kernelb = np.ones((3, 3), np.uint8)


# applique les transpho morphologiques à l'image
def morph_trans(ima):
    global kernel, kernelb
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)  # clustering
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernelb)  # denoise
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # denoise
    # ima = cv.dilate(ima, np.ones((5, 5), np.uint8), iterations=1)
    return ima


def process(q_to, q_has):
    while True:
        get = q_to.get()
        frame = get[0]
        frame_treated = get[1]
        if get[2]:
            frame_treated = seg_hsv(frame)
            frame_treated = morph_trans(frame_treated)
        q_has.put((frame, frame_treated, get[2]))
