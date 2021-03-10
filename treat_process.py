import cv2 as cv
import numpy as np
import methods as meth

to = False  # global
meansat = 0  # temp


# segmentation (HSV)
def seg_hsv(img):
    global to, meansat
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # meth.display_hist(img,count)
    h, s, v = cv.split(img)
    meansat = np.mean(s)
    if np.mean(h) > 30 or np.mean(s) < 50:  # bri
        to = False
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
    global to, meansat
    while True:
        get = q_to.get()
        frame = get[0]
        frame_treated = get[1]
        to = get[2]
        if to:
            frame_treated = seg_hsv(frame)
            frame_treated = morph_trans(frame_treated)
        q_has.put((frame, frame_treated, to, get[3], get[4], meansat))
