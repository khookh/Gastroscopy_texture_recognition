import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.filters import rank as skr
from skimage.morphology import disk


def norm_hist(ima):
    histogram, bin_edges = np.histogram(
        ima.flatten(), bins=256, range=(0, 256)
    )
    return 1. * histogram / np.sum(histogram)
    # hist, bins = np.histogram(ima.flatten(), range(256))  # histogram is computed on a 1D distribution --> flatten()
    # return 1. * hist / np.sum(hist)  # normalized histogram


def display_hist(ima, vmin=None, vmax=None):
    figure = plt.figure(figsize=[10, 5])
    if ima.ndim == 2:
        nh = norm_hist(ima)
    else:
        nh_r = norm_hist(ima[:, :, 0])
        nh_g = norm_hist(ima[:, :, 1])
        nh_b = norm_hist(ima[:, :, 2])
    # display the results
    plt.subplot(1, 2, 1)
    plt.imshow(ima, cmap=cm.gray, vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2)
    if ima.ndim == 2:
        plt.plot(nh, label='hist.')
    else:
        plt.plot(nh_r, color='r', label='r')
        plt.plot(nh_g, color='g', label='g')
        plt.plot(nh_b, color='b', label='b')
        plt.savefig('hist.png')
    plt.legend()
    plt.xlabel('gray level');
    plt.show()
    return plt


def apply_lut(ima, lut, vmin=None, vmax=None):
    nh = norm_hist(ima)
    lima = lut[ima]
    nh_lima = norm_hist(lima)

    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.imshow(lima / 255, cmap=cm.gray, vmin=vmin, vmax=vmax)
    ax1 = plt.subplot(1, 2, 2)
    plt.plot(nh, label='ima')
    plt.plot(nh_lima, label='lut[ima]')
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    plt.plot(lut, label='lut', color='k')
    plt.legend()
    plt.show()


def lut_autolevel(ima):
    g_min = np.min(ima)
    g_max = np.max(ima)
    lut = 255 * (np.arange(0, 256) - g_min) / (1. * g_max - g_min)
    return lut


def lut_equalization(ima):
    nh = norm_hist(ima)
    ch = np.append(np.array(0), np.cumsum(nh))
    lut = 255 * ch
    return lut


def shift_sobel(img):
    # cv.pyrMeanShiftFiltering(img, 10, 20, img)
    sobel_img = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=1)
    # cv.imwrite('sobel.png',sobel_img)
    # methods.display_hist(sobel_img)
    return sobel_img


def apply_medium_filter(ima):
    m = skr.median(ima, disk(2))
    d = m - ima
    return d


# renvoie une image segmentée par histogramme
def seg_his(ima, path):
    # methods.display_hist(ima)
    t = ima > 240
    # plt.imshow(t, cmap=plt.cm.gray)
    # plt.show()
    sel = np.zeros_like(ima)
    sel[t] = ima[t]
    # viewer = skimage.viewer.ImageViewer(sel)
    # viewer.show()
    cv.imwrite(path, sel)
    return sel
