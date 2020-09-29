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

# img_gastro = mpimg.imread('bulles.png')
# imgplot = plt.imshow(img_gastro)
# plt.show()

image_gastro = skimage.io.imread('bulles.png')
viewer = skimage.viewer.ImageViewer(image_gastro)
# viewer.show()

#display basic hist of the picture
#methods.display_hist(image_gastro)

#negative trans + hist
#lut = np.arange(255, -1, -1)
#methods.apply_lut(image_gastro, lut)

#auto level
#t_ima = (image_gastro/4+64).astype(np.uint8)
#methods.display_hist(t_ima,vmin=0,vmax=255)
#methods.apply_lut(t_ima,methods.lut_autolevel(t_ima))

#equalization
#methods.apply_lut(image_gastro,methods.lut_equalization(image_gastro))

#equalization in a certain square
roi = [(300,700),300,200] #around the bubles
sample = image_gastro[roi[0][1]:roi[0][1]+roi[2],roi[0][0]:roi[0][0]+roi[1]]
lut = methods.lut_equalization(sample)

plt.figure()
plt.imshow(image_gastro/255,cmap=cm.gray)
rect = plt.Rectangle(*roi, facecolor=None,alpha=.25)
plt.gca().add_patch(rect)
plt.figure()

plt.imshow(lut[image_gastro]/255,cmap=cm.gray);
rect = plt.Rectangle(*roi, facecolor=None,alpha=.25)
plt.gca().add_patch(rect);
plt.show()