#!/usr/bin/python
# for test purpose
import sys
import numpy as np
import skimage.color
import skimage.viewer
import cv2 as cv
import matplotlib.pyplot as plt
import methods as meth
from skimage.feature import greycomatrix, greycoprops


def texture_descriptor(N):
    displacement = 25
    angles = [0, np.pi / 6, np.pi / 4, np.pi / 3]
    glcm = greycomatrix(N, [displacement], angles, 256)
    return greycoprops(glcm, 'dissimilarity').max()


def sliding_window_overlap(im, PATCH_SIZE, STRIDE):
    output = np.zeros((im.shape[0], im.shape[1]))
    for i in range(0, im.shape[0] - PATCH_SIZE[0] + 1, STRIDE):
        for j in range(0, im.shape[1] - PATCH_SIZE[1] + 1, STRIDE):
            patch = im[i:i + PATCH_SIZE[0], j:j + PATCH_SIZE[1]]
            c = (i + PATCH_SIZE[0] // 2, j + PATCH_SIZE[1] // 2)  # center of the patch
            output[c[0] - STRIDE:c[0] + STRIDE, c[1] - STRIDE:c[1] + STRIDE] = texture_descriptor(patch)
    return output


cap = cv.VideoCapture(str(sys.argv[1]))
count = 1
while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
    cap = cv.VideoCapture(str(sys.argv[1]));
    cv.waitKey(500)
    print("wait")
while True:
    ret, frame = cap.read()
    k = cv.waitKey(1) & 0xFF
    if k == ord('p'):
        while True:
            if cv.waitKey(1) & 0xFF == ord('s'):
                break
    if ret:
        frame = cv.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv.INTER_CUBIC)
        img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img)
        print(np.mean(h))

        cv.imshow('o',frame)
        count += 1
    else:
        break
cap.release()
cv.destroyAllWindows()
