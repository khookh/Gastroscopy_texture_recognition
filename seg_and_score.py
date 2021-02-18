#!/usr/bin/python

import numpy as np
import skimage.color
import skimage.viewer
import cv2 as cv
import queue
from threading import Thread
from multiprocessing import Process, Queue
import sys
import os
import thread_wrapper as t_w
import time
import treat_process as t_p
import methods as meth


# renvoie un score de qualité à partir de l'image binaire
def score(ima, _dim):
    scoring = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        scoring = bad_pixels.shape[0] / (_dim[0] * _dim[1])
    return scoring


kernel = np.ones((7, 7), np.uint8)
kernelb = np.ones((3, 3), np.uint8)


# returns the uniformity of the image
def uniformity(ima):
    blur1_uni = cv.GaussianBlur(ima, (5, 5), 1)
    blur2_uni = cv.GaussianBlur(ima, (31, 31), 2)
    return np.sum((blur1_uni - blur2_uni) ** 2)


# lecture flux vidéo
count = 1
over = False
q_frame = queue.Queue()  # thread

q_to_treat = Queue()  # process
q_treated = Queue()  # process

if str(sys.argv[3]) == "-usb":  # temporaire
    cap = cv.VideoCapture(0)
    wrap = t_w.Wrap_("output_hd")
else:
    cap = cv.VideoCapture(str(sys.argv[1]))
    wrap = t_w.Wrap_(os.path.basename(str(sys.argv[1])))


# Thread reading the video flux
def read_flux():
    global count, cap, over
    while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
        if str(sys.argv[3]) == "-usb":
            cap = cv.VideoCapture(0)
        else:
            cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(500)
    while over is False:
        ret, frame = cap.read()
        if ret:
            q_frame.put(cv.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC))
            count += 1
        else:
            over = True
            break
        if q_frame.qsize() > 100:
            time.sleep(0)
    cap.release()


# thread treating the frames
def frame_treatment():
    global count, wrap, over
    local_count = 1
    while over is False:
        if q_frame.empty():
            time.sleep(0)
        frame = q_frame.get()
        if local_count == 1:
            dimensions = frame.shape
            wrap.dim = dimensions  # temp
            frame_treated = np.zeros(dimensions)

        # uniformity
        unfy = uniformity(frame) / (wrap.dim[0] * wrap.dim[1])
        wrap.blur_list = np.append(wrap.blur_list, unfy)
        wrap.w_check()

        if local_count % int(sys.argv[4]) == 0:
            if unfy > 22 and wrap.p_capture is False:
                q_to_treat.put((frame, frame_treated, True))
            else:
                q_to_treat.put((frame, frame_treated, False))
        local_count += 1
        while q_treated.empty() is False: #pour assurer la synchro lors du save(), temp
            if over:
                break
            time.sleep(0)

    wrap.save()
    wrap.section_score()
    wrap.output_f(count)


# Thread displaying the frames
def display_t():
    global wrap, over, dim
    local_count = 1
    start = time.time()
    fps = 0
    while over is False:
        k = cv.waitKey(1) & 0xFF
        if k == ord('p'):
            while True:
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
        if k == ord('q'):
            over = True
        if q_treated.empty():
            if over:
                break
            time.sleep(0)

        source = q_treated.get()
        frame = source[0]
        frame_treated = source[1]
        if source[2]:
            wrap.temp_score_list = np.append(wrap.temp_score_list, round(score(frame_treated, wrap.dim) * 100, 3))
        else:
            wrap.save()
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
            if q_treated.get()[1] is None:
                break
            frame = np.hstack((frame, cv.cvtColor(q_treated.get()[1], cv.COLOR_GRAY2BGR)))
        frame = cv.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
        # rajoute les paramètres informatifs
        image = cv.putText(frame, 'Frame %d' % local_count, (5, 310), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255),
                           1,
                           cv.LINE_AA)
        # image = cv.putText(image, 'mean score = %.2f' % np.mean(wrap.section_score_list), (5, 290),
        image = cv.putText(image, 'dim = (%.2f,%2.f)' % (wrap.dim[0], wrap.dim[1]), (5, 100),
                           cv.FONT_HERSHEY_SIMPLEX, .5,
                           (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'fps = %.2f' % fps, (5, 220),
                           cv.FONT_HERSHEY_SIMPLEX, .5,
                           (0, 0, 255), 1,
                           cv.LINE_AA)

        cv.imshow('comparison', image)
        local_count += 1
        # cv.imwrite('frames/frame%d.png' % local_count, image)
    cv.destroyAllWindows()


if __name__ == '__main__':
    thread_fetch = Thread(target=read_flux)
    thread_treatment = Thread(target=frame_treatment)
    thread_display = Thread(target=display_t)
    thread_fetch.start()
    thread_treatment.start()
    thread_display.start()
    treat_proc = Process(target=t_p.process, args=(q_to_treat, q_treated,))
    treat_proc.daemon = True
    treat_proc.start()
    thread_treatment.join()
    treat_proc.kill()

# thread treatment stops when either the display or the fetch has stopped
# TODO : refactor thread managing -> multiprocessing ?
# TODO : delelete unecessary calls
