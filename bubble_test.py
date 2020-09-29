import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.data import camera, astronaut
from IPython.display import HTML, Image, SVG, YouTubeVideo

img_gastro = mpimg.imread('bulles.png')
imgplot = plt.imshow(img_gastro)
plt.show()

plt.style.use('ggplot')

def norm_hist(ima):
    hist, bins = np.histogram(ima.flatten(), range(256))  # histogram is computed on a 1D distribution --> flatten()
    return 1. * hist / np.sum(hist)  # normalized histogram


def display_hist(ima, vmin=None, vmax=None):
    plt.figure(figsize=[10, 5])
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
    plt.legend()
    plt.xlabel('gray level');
    plt.show()

display_hist(img_gastro)


def apply_lut(ima, lut, vmin=None, vmax=None):
    nh = norm_hist(ima)
    lima = lut[ima]
    nh_lima = norm_hist(lima)

    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.imshow(lima, cmap=cm.gray, vmin=vmin, vmax=vmax)
    ax1 = plt.subplot(1, 2, 2)
    plt.plot(nh, label='ima')
    plt.plot(nh_lima, label='lut[ima]')
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    plt.plot(lut, label='lut', color='k')
    plt.legend()
