import numpy as np
import skimage
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.data import camera, astronaut
import skimage.color
import skimage.io
import skimage.viewer
from IPython.display import HTML, Image, SVG, YouTubeVideo
def norm_hist(ima):
    histogram, bin_edges = np.histogram(
        ima.flatten(), bins=256, range=(0, 256)
    )
    return 1. * histogram / np.sum(histogram)
    #hist, bins = np.histogram(ima.flatten(), range(256))  # histogram is computed on a 1D distribution --> flatten()
    #return 1. * hist / np.sum(hist)  # normalized histogram


def display_hist(ima, vmin=None, vmax=None):
    plt.figure(figsize=[10, 5])
    nh_r = norm_hist(ima[:, :, 0])
    nh_g = norm_hist(ima[:, :, 1])
    nh_b = norm_hist(ima[:, :, 2])
    # display the results
    plt.subplot(1, 2, 1)
    plt.imshow(ima, cmap=cm.gray, vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2)
    plt.plot(nh_r, color='r', label='r')
    plt.plot(nh_g, color='g', label='g')
    plt.plot(nh_b, color='b', label='b')
    plt.legend()
    plt.xlabel('gray level');
    plt.show()
def apply_lut(ima, lut, vmin=None, vmax=None):
    nh = norm_hist(ima)
    lima = lut[ima]
    nh_lima = norm_hist(lima)

    plt.figure(figsize=[10, 5])
    plt.subplot(1, 2, 1)
    plt.imshow(lima/255, cmap=cm.gray, vmin=vmin, vmax=vmax)
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
    lut = 255*(np.arange(0,256)-g_min)/(1.*g_max-g_min)
    return lut
def lut_equalization(ima):
    nh = norm_hist(ima)
    ch = np.append(np.array(0),np.cumsum(nh))
    lut = 255*ch
    return lut
