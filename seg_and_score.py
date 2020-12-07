#!/usr/bin/python

import numpy as np
import skimage.color
import skimage.viewer
import cv2 as cv
import queue
from threading import Thread
import time
import sys
import os


# segmentation (HSV)
def seg_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    mean_sat = np.mean(np.ndarray.flatten(s))
    # temp seg masks
    mask = cv.inRange(img, (0, 35, 170), (60, 100, 245))  # direct light
    mask2 = cv.inRange(img, (0, 0, 90), (30, 95, 170))  # low light foam
    return mask + mask2, mean_sat
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
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)  # clustering
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))  # denoise
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # denoise
    # ima = cv.dilate(ima, np.ones((5, 5), np.uint8), iterations=1)
    return ima


def strict_diff():
    global blur_list, count
    if blur_list.size > 4:
        if blur_list[-1] != blur_list[-2] and blur_list[-2] != blur_list[-3] and blur_list[-3] != blur_list[-4]:
            return True
    return False


def strict_eq():
    global blur_list, count
    if blur_list.size > 4:
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
section, count = 1, 1
mean_s, sco, blur = 0, 0, 0
pause = False
p_capture = False
over = False
blur_list = np.array([])
score_list = np.array([])
temp_score_list = np.array([])
section_score_list = np.array([])
q_frame = queue.Queue()
q_treated = queue.Queue()
f = open("output_%s.txt" % os.path.basename(str(sys.argv[1])), "w")


def read_flux():
    global count, pause, over
    cap = cv.VideoCapture(str(sys.argv[1]))
    while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
        pause = True
        cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(500)
    while cap.isOpened():
        while q_frame.qsize() > 50:
            time.sleep(0)
        retr, frame = cap.read()
        q_frame.put(frame)
        if retr is not True and count > 1:
            cap.release()
            over = True
            break
        elif retr is True:
            count += 1
        if over is True:
            cap.release()
            break


def frame_treatment():
    global temp_score_list, section_score_list, score_list, blur_list, count, section, p_capture, blur, mean_s, pause
    local_count = 1
    while True:
        while pause is True and q_frame.empty():
            cv.waitKey(1)
        frame = q_frame.get()
        if local_count == 1:
            dimensions = frame.shape
            centrex, centrey = dimensions[1] / 2, dimensions[0] / 2
            dim = (int(centrex), int(centrey))
            frame_treated = np.zeros(dimensions)
        # motion blur level
        blur = cv.Laplacian(frame, cv.CV_32F).var()
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
                temp_score_list = np.append(temp_score_list, round(score(frame_treated, dim) * 100, 3))
            else:
                save()
        else:
            save()
        if over is True:
            save()
            section_score()
            break
        q_treated.put(frame)
        local_count += 1


def display_t():
    global score_list, count, pause, over
    local_count = 1
    while True:
        while pause is True and q_treated.empty():
            cv.waitKey(1)
        if q_treated.empty():
            time.sleep(0)
        frame = q_treated.get()
        # Affichage
        frame = skimage.color.gray2rgb(frame)
        # resize pour affichage propre
        # frame_treated_f = cv.resize(frame_treated_f, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        frame = cv.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        # concatene les deux images pour comparaison
        # numpy_h_concat = np.hstack((frame, frame_treated_f))
        # rajoute les paramètres informatifs
        image = cv.putText(frame, 'Frame %d' % local_count, (5, 370), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255),
                           1,
                           cv.LINE_AA)
        image = cv.putText(image, 'mean score = %.2f' % np.mean(section_score_list), (5, 400),
                           cv.FONT_HERSHEY_SIMPLEX, .5,
                           (0, 0, 255), 1,
                           cv.LINE_AA)
        # image = cv.putText(image, 'msat = %d' % round(mean_s, 3), (5, 420), cv.FONT_HERSHEY_SIMPLEX, .5,
        #                   (0, 0, 255), 1,
        #                   cv.LINE_AA)
        # image = cv.putText(image, 'blur = %d' % round(blur), (5, 350), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
        #                   cv.LINE_AA)
        # show dans la fenêtre
        cv.imshow('comparison', image)
        local_count+=1
        # cv.imwrite('frames/frame%d.png' % count, image)
        k = cv.waitKey(1) & 0xFF
        if k == ord('p'):
            pause = True
            while True:
                if cv.waitKey(1) & 0xFF == ord('s'):
                    pause = False
                    break
        elif k == ord('q'):
            over = True
            cv.destroyAllWindows()
            break


thread_fetch = Thread(target=read_flux)
thread_treatment = Thread(target=frame_treatment)
thread_display = Thread(target=display_t)
thread_fetch.start()
thread_treatment.start()
thread_display.start()
thread_fetch.join()
thread_treatment.join()
thread_display.join()

f.write("Mean score of whole video = %.2f \n" % np.mean(score_list))
f.write("(%.2f %% of the frame from the video were treated)" % (score_list.size * 100.0 / count))
f.close()
