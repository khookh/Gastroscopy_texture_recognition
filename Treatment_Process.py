import cv2 as cv
import numpy as np
import Score_Wrapper as t_w
import time

to = False  # global


def score(ima, _dim):
    """
    :param ima: mask of detected pollution
    :param _dim: dimension of the mask/frame
    :return: density of pollution
    """
    scoring = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        scoring = bad_pixels.shape[0] / (_dim[0] * _dim[1])
    return scoring


def uniformity(ima):
    """
    Measure the 'uniformity' (or blur effect) of the given frame
    :param ima: input frame
    :return: uniformity score
    """
    blur1_uni = cv.GaussianBlur(ima, (5, 5), 1)
    blur2_uni = cv.GaussianBlur(ima, (31, 31), 2)
    return np.sum((blur1_uni - blur2_uni) ** 2)


def seg_hsv(img,wrap):
    """
    Transform input into corresponding HSV color space and isolate parts of this space corresponding to pollution
    :param wrap:
    :param img: frame to be segmented
    :return: mask of detected pollution
    """
    global to
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img)
    if np.mean(h) > 15 or np.mean(s) < 65:  # threshold to remove frames affected by blue light or too close from lamp
        to = False
    # temp seg masks
    if wrap.section == 1:
        return cv.inRange(img, (0, 20, 170), (60, 60, 245))
    else:
        mask = cv.inRange(img, (0, 35, 170), (60, 100, 245))  # direct light
        mask2 = cv.inRange(img, (0, 0, 90), (30, 80, 170))  # low light foam
        return mask + mask2


# kernel used for morphological transform
kernel = np.ones((7, 7), np.uint8)
kernelb = np.ones((3, 3), np.uint8)


def morph_trans(ima):
    """
    Apply morphological transforms to the frame in order to : suppress (amap) noise + fill detected areas
    :param ima: frame to be transformed
    :return: transformed framed
    """
    global kernel, kernelb
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)  # clustering
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernelb)  # de-noise
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # de-noise
    return ima


def process(q_frame, q_treated, path, v):
    """
    Process responsible for frame treatment (pollution detection + DNN detection)
    :param q_frame: frames fetched the thread read_flux
    :param q_treated: frames ready to be displayed
    :param path: path of the file/source
    :param v: end process flag
    """
    wrap = t_w.Wrap_(path)
    global to
    local_count = 1
    while True:
        while q_frame.empty():  # if no more frame available for processing
            if v.value == 1:  # if end process flag is on
                break
            time.sleep(0)
        if v.value == 1:  # if end process flag is on
            break
        frame = q_frame.get()  # get frame from queue
        if local_count == 1:
            wrap.dim = frame.shape
        unfy = uniformity(frame) / (wrap.dim[0] * wrap.dim[1])
        wrap.uniformity_list = np.append(wrap.uniformity_list, unfy)
        wrap.w_check(frame)
        if unfy > 20 and wrap.p_capture is False:
            to = True
            frame_treated = seg_hsv(frame,wrap)
            if to:
                frame_treated = morph_trans(frame_treated)
                wrap.temp_score_list = np.append(wrap.temp_score_list, round(score(frame_treated, wrap.dim) * 100, 3))
            else:
                wrap.save()
                frame_treated = np.zeros(wrap.dim)
        else:
            frame_treated = np.zeros(wrap.dim)
        q_treated.put((frame, frame_treated, np.mean(wrap.section_score_list), unfy))
        local_count += 1

    wrap.save()
    wrap.section_score()
    wrap.output_f(local_count)
    cv.destroyAllWindows()
