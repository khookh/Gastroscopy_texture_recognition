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
import thread_wrapper as t_w
import time
import methods as meth


# segmentation (HSV)
def seg_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    meth.display_hist(img,count)
    h, s, v = cv.split(img)
    # temp seg masks
    mask = cv.inRange(img, (0, 35, 170), (60, 100, 245))  # direct light
    mask2 = cv.inRange(img, (0, 0, 90), (30, 95, 170))  # low light foam
    return mask + mask2
    # return cv.bitwise_and(img, img, mask=mask)


# renvoie un score de qualité à partir de l'image binaire
def score(ima, _dim):
    scoring = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        scoring = bad_pixels.shape[0] / (4 * _dim[0] * _dim[1])
    return scoring


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


# returns the uniformity of the image
def uniformity(ima):
    blur1_uni = cv.GaussianBlur(ima, (5, 5), 1)
    blur2_uni = cv.GaussianBlur(ima, (31, 31), 2)
    return np.sum((blur1_uni - blur2_uni) ** 2)


# lecture flux vidéo
count = 1
over = False
q_frame = queue.Queue()
q_treated = queue.Queue()

if str(sys.argv[3]) == "-usb":  # temporaire
    cap = cv.VideoCapture(0)
    wrap = t_w.Wrap_("output_hd")
else:
    cap = cv.VideoCapture(str(sys.argv[1]))
    wrap = t_w.Wrap_(os.path.basename(str(sys.argv[1])))


# Thread reading the video flux
def read_flux():
    global count, over, cap
    while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
        if str(sys.argv[3]) == "-usb":
            cap = cv.VideoCapture(0)
        else:
            cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(500)
    while True:
        retr, frame = cap.read()
        if over is True:
            cap.release()
            cv.destroyAllWindows()
            over = True
            print("read stop \n")
            break
        if retr:
            q_frame.put(cv.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC))
            count += 1
        else:
            cap.release()
            cv.destroyAllWindows()
            over = True
            break
        if q_frame.qsize() > 100:
            time.sleep(0)

    cap.release()
    cv.destroyAllWindows()
    over = True

    print("read stop \n")


# thread treating the frames
def frame_treatment():
    global count, over, wrap
    local_count = 1
    dim = (0, 0)
    while True:
        if over is True:
            cv.destroyAllWindows()
            wrap.save()
            wrap.section_score()
            wrap.output_f(count)
            print("treatment stop \n")
            break
        if q_frame.empty() and over is False:
            time.sleep(0)
        frame = q_frame.get()
        if local_count == 1:
            dimensions = frame.shape
            wrap.dim = dimensions  # temp
            centrex, centrey = dimensions[1] / 2, dimensions[0] / 2
            dim = (int(centrex), int(centrey))
            frame_treated = np.zeros(dimensions)
        # uniformity
        unfy = uniformity(frame) / (dim[0] * dim[1] * 4)
        wrap.blur_list = np.append(wrap.blur_list, unfy)
        wrap.w_check()

        if local_count % int(sys.argv[4]) == 0:
            if unfy > 22 and wrap.p_capture is False:
                frame_treated = seg_hsv(frame)
                frame_treated = morph_trans(frame_treated)
                wrap.temp_score_list = np.append(wrap.temp_score_list, round(score(frame_treated, dim) * 100, 3))
            else:
                wrap.save()
        q_treated.put((frame, frame_treated))
        local_count += 1


# Thread displaying the frames
def display_t():
    global wrap, count, over
    local_count = 1
    start = time.time()
    fps = 0
    while True:
        k = cv.waitKey(1) & 0xFF
        if k == ord('p'):
            while True:
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
        if k == ord('q') or over is True:
            over = True
            cv.destroyAllWindows()
            print("disp stop \n")
            break
        if q_treated.empty():
            time.sleep(0)
        frame = q_treated.get()[0]
        # fps
        if local_count % 40 == 0:
            end = time.time()
            elapsed = (end - start)
            fps = round(40 / elapsed)
            wrap.fps_list = np.append(wrap.fps_list, fps)
            start = end
        # Affichage
        frame = skimage.color.gray2rgb(frame)
        # resize pour affichage propre
        # concatene les deux images pour comparaison
        if str(sys.argv[2]) == "-conc":  # temporaire
            frame = np.hstack((frame, skimage.color.gray2rgb(q_treated.get()[1])))
            frame = cv.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
        # rajoute les paramètres informatifs
        image = cv.putText(frame, 'Frame %d' % local_count, (5, 310), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255),
                           1,
                           cv.LINE_AA)
        # image = cv.putText(image, 'mean score = %.2f' % np.mean(wrap.section_score_list), (5, 290),
        image = cv.putText(image, 'dim = (%.2f,%2.f)' % (wrap.dim[0],wrap.dim[1]), (5, 100),
                           cv.FONT_HERSHEY_SIMPLEX, .5,
                           (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'fps = %.2f' % fps, (5, 220),
                           cv.FONT_HERSHEY_SIMPLEX, .5,
                           (0, 0, 255), 1,
                           cv.LINE_AA)


        #cv.imshow('comparison', image)
        local_count += 1
        cv.imwrite('frames/frame%d.png' % local_count, image)


thread_fetch = Thread(target=read_flux)
thread_treatment = Thread(target=frame_treatment)
thread_display = Thread(target=display_t)
thread_fetch.start()
thread_treatment.start()
thread_display.start()
# thread treatment stops when either the display or the fetch has stopped


# TODO : refactor thread managing
# TODO : delelete unecessary calls
