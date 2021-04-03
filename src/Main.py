#!/usr/bin/python

import numpy as np
import skimage.color
import skimage.viewer
import cv2 as cv
from threading import Thread
from multiprocessing import Process, Queue, Value
import sys
import os
import time
from src import Treatment_Process as t_p, Score_Wrapper as t_w


def read_flux():
    """
    Thread fetching frames
    """
    global count, cap, over
    ratio = 1
    while not cap.isOpened():  # active waiting if the input isn't ready yet (for real-time)
        if str(sys.argv[3]) == "-usb":
            cap = cv.VideoCapture(0)
        else:
            cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(500)
    while over is False:
        ret, frame = cap.read()
        while q_frame.qsize() > 100:  # to avoid overusing too many resources
            time.sleep(0)
        if ret:
            if count == 1:
                ratio = round(frame.shape[0] / 216, 2)  # ratio to reduce the size of the image to 216 in height
                print(ratio)
            q_frame.put(cv.resize(frame, None, fx=1 / ratio, fy=1 / ratio, interpolation=cv.INTER_CUBIC))
            count += 1
        else:
            over = True
            break
    cap.release()


def info_on_frame(source, local_count, fps):
    """
    Prepare the frames and information in order to display them as required
    :param source: frames and important information
    :param local_count: number of the frame to display
    :param fps: measured fps count
    :return: image to show
    """
    frame = source[0]
    frame_treated = source[1]

    # Affichage
    frame = skimage.color.gray2rgb(frame)
    # resize pour affichage propre
    # concatene les deux images pour comparaison
    if str(sys.argv[2]) == "-conc":  # temporaire
        if len(frame_treated.shape) == 2:
            frame = np.hstack((frame, cv.cvtColor(frame_treated, cv.COLOR_GRAY2RGB)))
    frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    # rajoute les paramètres informatifs
    image = cv.putText(frame, 'Frame %d ' % local_count, (5, 170), cv.FONT_HERSHEY_SIMPLEX, .4,
                       (0, 0, 255),
                       1,
                       cv.LINE_AA)
    image = cv.putText(image, 'mean score = %.2f' % source[2], (5, 130),
                       cv.FONT_HERSHEY_SIMPLEX, .5,
                       (0, 0, 255), 1,
                       cv.LINE_AA)
    image = cv.putText(image, 'unfy = %.2f' % source[3], (5, 80),
                       cv.FONT_HERSHEY_SIMPLEX, .5,
                       (0, 0, 255), 1,
                       cv.LINE_AA)
    image = cv.putText(image, 'fps = %.2f' % fps, (5, 220),
                       cv.FONT_HERSHEY_SIMPLEX, .5,
                       (0, 0, 255), 1,
                       cv.LINE_AA)
    return image


def display_t():
    """
    Thread displaying the frames in a window
    """
    global wrap, over
    local_count = 1
    start = time.time()
    fps = 0
    while True:
        k = cv.waitKey(1) & 0xFF
        if k == ord('p'):
            while True:
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
        if k == ord('q'):
            over = True
            break
        if q_treated.empty():
            if over and q_frame.empty():
                break
            time.sleep(0)
        else:
            source = q_treated.get()  # from treatment process

            # fps measured every 40 frames
            if local_count % 40 == 0:
                end = time.time()
                elapsed = (end - start)
                fps = round(40 / elapsed)
                start = end

            image = info_on_frame(source, local_count, fps)

            cv.imshow('comparison', image)
            #cv.imwrite('frames/test%d.png' % local_count, image)
            local_count += 1
    cv.destroyAllWindows()


if __name__ == '__main__':
    count = 1
    over = False
    q_frame = Queue()
    q_treated = Queue()

    if str(sys.argv[3]) == "-usb":  # temporaire
        cap = cv.VideoCapture(0)
        wrap = t_w.Wrap_("output_hd")
    else:
        cap = cv.VideoCapture(str(sys.argv[1]))
        wrap = t_w.Wrap_(os.path.basename(str(sys.argv[1])))

    thread_fetch = Thread(target=read_flux)
    thread_display = Thread(target=display_t)
    thread_fetch.start()
    thread_display.start()
    v = Value('i', 0)  # end flag for treatment process
    treat_proc = Process(target=t_p.process, args=(q_frame, q_treated, os.path.basename(str(sys.argv[1])), v,))
    treat_proc.daemon = True
    treat_proc.start()
    thread_display.join()
    v.value = 1
