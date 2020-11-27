#!/usr/bin/python

import numpy as np
import skimage.color
import skimage.viewer
import cv2 as cv
import sys


def mean_hs(list_h, list_s):
    _count = 0
    _sum_h = 0
    _sum_s = 0
    it = 0
    x, y = list_h.shape
    for i in range(x):
        for j in range(y):
            it += 1
            if it % 20 == 0:
                _count += 1
                _sum_h += list_h[i, j]
                _sum_s += list_s[i, j]
    return _sum_h / _count, _sum_s / _count


# segmentation (HSV)
def seg_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    mean_s = np.mean(np.ndarray.flatten(s))
    if 120 > mean_s > 80:  # test
        tresh_s = mean_s * 0.9
    else:
        tresh_s = mean_s
    # temp seg masks
    mask = cv.inRange(img, (0, 0, 170), (179, tresh_s, 240))  # direct light
    mask2 = cv.inRange(img, (0, 0, 10), (30, 45, 140))  # low light foam
    return mask + mask2, mean_s
    # return cv.bitwise_and(img, img, mask=mask)


# renvoie un score de qualité à partir de l'image binaire
def score(ima, dim):
    score = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        score = bad_pixels.shape[0] / (4 * dim[0] * dim[1])
    return score


# applique les transpho morphologiques à l'image
def morph_trans(ima):
    kernel = np.ones((7, 7), np.uint8)
    kernelb = np.ones((5, 5), np.uint8)
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)  # clustering
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # denoise
    ima = cv.dilate(ima, kernelb, iterations=1)
    return ima


# lecture flux vidéo
cap = cv.VideoCapture(str(sys.argv[1]))
count = 1
sco = 'null'
score_list = np.array([])
while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
    cap = cv.VideoCapture(str(sys.argv[1]))
    cv.waitKey(1000)
    print("Wait for the header")
while cap.isOpened():
    retr, frame = cap.read()
    if count == 1:  # récupère les dimensions au début
        dimensions = frame.shape
        centrex, centrey = dimensions[1] / 2, dimensions[0] / 2
        dim = (int(centrex), int(centrey))
    if retr:
        # motion blur level
        blur = cv.Laplacian(frame, cv.CV_64F).var()
        ret, mean_s = seg_hsv(frame)
        if blur < 1150 and mean_s < 130:
            ret = morph_trans(ret)
            sco = str(round(score(ret, dim) * 100, 3))
            score_list = np.append(score_list, [sco])

        # Affichage
        ret = skimage.color.gray2rgb(ret)
        # resize pour affichage propre
        ret = cv.resize(ret, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        frame = cv.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        # concatene les deux images pour comparaison
        numpy_h_concat = np.hstack((frame, ret))
        # rajoute les paramètres informatifs
        image = cv.putText(numpy_h_concat, 'Frame %d' % count, (5, 370), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'score = %s' % sco, (5, 400), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'msat = %d' % round(mean_s, 3), (5, 420), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'blur = %d' % round(blur), (5, 350), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                           cv.LINE_AA)
        # show dans la fenêtre
        # cv.imshow('comparison', image)
        # cv.imwrite('hsv_seg/test_sue%d_e.png' % count, image)

        count += 1
    else:  # si la frame n'est pas prête
        cv.waitKey(1)
    k = cv.waitKey(1) & 0xFF
    if k == ord('p'):
        while True:
            if cv.waitKey(1) & 0xFF == ord('s'):
                break
    elif k == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

print(np.mean(score_list))
