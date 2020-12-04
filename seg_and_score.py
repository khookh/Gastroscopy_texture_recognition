#!/usr/bin/python

import numpy as np
import skimage.color
import skimage.viewer
import cv2 as cv
import sys
import os


# segmentation (HSV)
def seg_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    mean_s = np.mean(np.ndarray.flatten(s))
    # temp seg masks
    mask = cv.inRange(img, (0, 35, 170), (60, 100, 245))  # direct light
    mask2 = cv.inRange(img, (0, 0, 90), (30, 95, 170))  # low light foam
    mask25  = cv.inRange(img, (170, 0, 90), (179, 95, 170))
    return mask + mask2 + mask25, mean_s
    # return cv.bitwise_and(img, img, mask=mask)


# renvoie un score de qualité à partir de l'image binaire
def score(ima, _dim):
    score = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        score = bad_pixels.shape[0] / (4 * _dim[0] * _dim[1])
    return score


# applique les transpho morphologiques à l'image
def morph_trans(ima):
    kernel = np.ones((9, 9), np.uint8)
    kernelb = np.ones((5, 5), np.uint8)
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))  # denoise
    #ima = cv.morphologyEx(ima, cv.MORPH_CLOSE,  np.ones((3, 3), np.uint8))# clustering
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # denoise
    # ima = cv.dilate(ima, kernelb, iterations=1)
    return ima


def strict_diff():
    global blur_list, count
    if count > 4:
        if blur_list[-1] != blur_list[-2] and blur_list[-2] != blur_list[-3] and blur_list[-3] != blur_list[-4]:
            return True
    return False


def strict_eq():
    global blur_list, count
    if count > 4:
        if blur_list[-1] == blur_list[-2] and blur_list[-2] == blur_list[-3] and blur_list[-3] == blur_list[-4]:
            return True
    return False


def section_score():
    global f, score_list, section_score_list, section
    f.write("Mean score in section %i = %.2f \n" % (section, np.mean(section_score_list)))
    f.write("_____________________\n")
    score_list = np.append(score_list, section_score_list)
    section_score_list = np.array([])
    section += 1


def save():
    global section_score_list, temp_score_list
    if temp_score_list.size > 8:
        section_score_list = np.append(section_score_list, temp_score_list)
    temp_score_list = np.array([])


# lecture flux vidéo
cap = cv.VideoCapture(str(sys.argv[1]))
f = open("output_%s.txt" % os.path.basename(str(sys.argv[1])), "w")
section, count = 1, 1
mean_s, last_count, sco = 0, 0, 0
p_capture = False
blur_list = np.array([])
score_list = np.array([])
temp_score_list = np.array([])
section_score_list = np.array([])
while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
    cap = cv.VideoCapture(str(sys.argv[1]))
    cv.waitKey(1000)
    print("Wait for the header")
while cap.isOpened():
    retr, frame = cap.read()
    if retr:
        if count == 1:
            dimensions = frame.shape
            centrex, centrey = dimensions[1] / 2, dimensions[0] / 2
            dim = (int(centrex), int(centrey))
            frame_treated = np.zeros(dimensions)
        # motion blur level
        blur = cv.Laplacian(frame, cv.CV_64F).var()
        blur_list = np.append(blur_list, round(blur))
        if strict_eq():
            p_capture = True
        if p_capture is True and strict_diff():
            p_capture = False
            section_score()
        if blur < 1200:
            frame_treated, mean_s = seg_hsv(frame)
            if mean_s < 130:  # np.mean(mean_sv) + 50:
                frame_treated = morph_trans(frame_treated)
                sco = (round(score(frame_treated, dim) * 100, 3))
                temp_score_list = np.append(temp_score_list, sco)

        else:
            save()
        try:
            # Affichage
            frame_treated_f = skimage.color.gray2rgb(frame_treated)
            # resize pour affichage propre
            frame_treated_f = cv.resize(frame_treated_f, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
            frame = cv.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
            # concatene les deux images pour comparaison
            numpy_h_concat = np.hstack((frame, frame_treated_f))
            # rajoute les paramètres informatifs
            image = cv.putText(numpy_h_concat, 'Frame %d' % count, (5, 370), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255),
                               1,
                               cv.LINE_AA)
            image = cv.putText(image, 'mean score = %.2f' % np.mean(section_score_list), (5, 400),
                               cv.FONT_HERSHEY_SIMPLEX, .5,
                               (0, 0, 255), 1,
                               cv.LINE_AA)
            image = cv.putText(image, 'msat = %d' % round(mean_s, 3), (5, 420), cv.FONT_HERSHEY_SIMPLEX, .5,
                               (0, 0, 255), 1,
                               cv.LINE_AA)
            image = cv.putText(image, 'blur = %d' % round(blur), (5, 350), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                               cv.LINE_AA)
            # show dans la fenêtre
            cv.imshow('comparison', image)
            # cv.imwrite('frames/frame%d.png' % count, image)
        except:
            print("Fail to display frame %d" % count)
        count += 1
    elif count != 1:  # si la frame n'est pas prête
        cv.waitKey(1)
        break
    k = cv.waitKey(1) & 0xFF
    if k == ord('p'):
        while True:
            if cv.waitKey(1) & 0xFF == ord('s'):
                break
    elif k == ord('q'):
        break
save()
section_score()
cap.release()
cv.destroyAllWindows()

f.write("Mean score of whole video = %.2f \n" % np.mean(score_list))
f.write("(%.2f %% of the frame from the video were treated)" % (score_list.size * 100.0 / count))
f.close()
