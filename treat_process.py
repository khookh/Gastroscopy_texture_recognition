import cv2 as cv
import numpy as np
import thread_wrapper as t_w
import time

to = False  # global


# renvoie un score de qualité à partir de l'image binaire
def score(ima, _dim):
    scoring = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        scoring = bad_pixels.shape[0] / (_dim[0] * _dim[1])
    return scoring


# returns the uniformity of the image
def uniformity(ima):
    blur1_uni = cv.GaussianBlur(ima, (5, 5), 1)
    blur2_uni = cv.GaussianBlur(ima, (31, 31), 2)
    return np.sum((blur1_uni - blur2_uni) ** 2)


# segmentation (HSV)
def seg_hsv(img):
    global to
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    if np.mean(h) > 30 or np.mean(s) < 75:  # bri
        to = False
    # temp seg masks
    mask = cv.inRange(img, (0, 35, 170), (60, 100, 245))  # direct light
    mask2 = cv.inRange(img, (0, 0, 90), (30, 70, 170))  # low light foam
    return mask + mask2


kernel = np.ones((7, 7), np.uint8)
kernelb = np.ones((3, 3), np.uint8)


# applique les transpho morphologiques à l'image
def morph_trans(ima):
    global kernel, kernelb
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)  # clustering
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernelb)  # denoise
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # denoise
    return ima


def process(q_frame, q_treated, path):
    wrap = t_w.Wrap_(path)
    global to
    local_count = 1
    while True:
        while q_frame.empty():
            time.sleep(0)
        frame = q_frame.get()
        if local_count == 1:
            wrap.dim = frame.shape
        unfy = uniformity(frame) / (wrap.dim[0] * wrap.dim[1])
        wrap.uniformity_list = np.append(wrap.uniformity_list, unfy)
        wrap.w_check(frame)
        if unfy > 22 and wrap.p_capture is False:
            to = True
            frame_treated = seg_hsv(frame)
            if to:
                frame_treated = morph_trans(frame_treated)
                wrap.temp_score_list = np.append(wrap.temp_score_list, round(score(frame_treated, wrap.dim) * 100, 3))
            else:
                wrap.save()
                frame_treated = np.zeros(wrap.dim)
        else:
            frame_treated = np.zeros(wrap.dim)
        q_treated.put((frame, frame_treated,np.mean(wrap.section_score_list),unfy))
        local_count += 1
    wrap.save()
    wrap.section_score()
    wrap.output_f(count)
