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
import methods
from skimage.io import imsave

# img_gastro = mpimg.imread('bulles.png')
# imgplot = plt.imshow(img_gastro)
# plt.show()

image_gastro = skimage.io.imread('bulles.png')
viewer = skimage.viewer.ImageViewer(image_gastro)
# viewer.show()

# display basic hist of the picture
# methods.display_hist(image_gastro)

# negative trans + hist
# lut = np.arange(255, -1, -1)
# methods.apply_lut(image_gastro, lut)

# auto level
# t_ima = (image_gastro/4+64).astype(np.uint8)
# methods.display_hist(t_ima,vmin=0,vmax=255)
# methods.apply_lut(t_ima,methods.lut_autolevel(t_ima))

# equalization
# methods.apply_lut(image_gastro,methods.lut_equalization(image_gastro))

# equalization in a certain square
# roi = [(800,200),300,200]
roi = [(300, 750), 300, 200]  # around the bubles
sample = image_gastro[roi[0][1]:roi[0][1] + roi[2], roi[0][0]:roi[0][0] + roi[1]]
lut = methods.lut_equalization(sample)

plt.figure()
plt.imshow(image_gastro / 255, cmap=cm.gray)
rect = plt.Rectangle(*roi, facecolor=None, alpha=.25)
plt.gca().add_patch(rect)
plt.figure()

plt.imshow(lut[image_gastro] / 255, cmap=cm.gray);
rect = plt.Rectangle(*roi, facecolor=None, alpha=.25)
plt.gca().add_patch(rect)
plt.show()
imsave('local_equalization.png',lut[image_gastro])

##convolution
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.cm as cm
import scipy as scp
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from mpl_toolkits.mplot3d import axes3d
from skimage.data import camera
from skimage.filters import rank as skr
from skimage.morphology import disk

im = skimage.color.rgb2gray(lut[image_gastro])

w, h = im.shape

n = 32

# square filter
s = np.zeros(im.shape, dtype=np.complexfloating)
s[int(w / 2) - n:int(w / 2) + n, int(h / 2) - n:int(h / 2) + n] = 1.0 + 0.0j

# circular filter
c = np.zeros(im.shape, dtype=np.complexfloating)
for i in range(w):
    for j in range(h):
        if ((i - w / 2) ** 2 + (j - h / 2) ** 2) < (n * n):
            c[i, j] = 1.0 + 0.0j

# smooth filter borders
c = skr.mean(np.real(c * 255).astype('uint8'), disk(10))
c = c.astype(np.complexfloating) / (255.0 + 0j)

F1 = fft2(im.astype(np.complexfloating))
F3 = F1 * ifftshift(s)
F4 = F1 * ifftshift(c)

# high pass using the complement of c
F5 = F1 * ifftshift((1.0 + 0j) - c)

psF1 = (F1 ** 2).real

low_pass_rec = ifft2(F3)
low_pass_circ = ifft2(F4)
high_pass_circ = ifft2(F5)

fig = plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(1.0 - c.real, interpolation='nearest', origin='upper')
plt.title('$g(x,y)')
plt.subplot(1, 2, 2)
plt.imshow(high_pass_circ.real, interpolation='nearest', origin='upper', cmap=cm.gray)
plt.title('$g(x,y)*f(x,y)$');
plt.show()
imsave('conv.png',high_pass_circ.real)
