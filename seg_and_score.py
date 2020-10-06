import numpy as np
import skimage.color
import skimage.io
import skimage.viewer
import methods
import cv2 as cv

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
    # cv.imwrite(path, sel)
    return sel

# renvoie un score de qualité à partir de l'image segmentée
def score(ima, dim):
    liste_defauts = cv.findNonZero(ima)
    # prototype
    dmax = np.math.sqrt(dim[1] * dim[1] + dim[0] * dim[0])
    score = 0
    if liste_defauts !=  None:
        for elem in liste_defauts:
            difx = elem[0][0] - dim[0]
            dify = elem[0][1] - dim[1]
            dnorm = (np.math.sqrt(difx * difx + dify * dify)) / dmax
            if dnorm > 1:
                print(dnorm)
            score += (1.7 / 5000) * (1 - pow(dnorm,3)) #coef arbitraire basé sur l'exp
        if score > 15:
            score = 15
        print(score)
    return score

# test
image_gastro = skimage.io.imread('bulles.png')
methods.display_hist(image_gastro)
image_gastro = skimage.util.img_as_ubyte(skimage.color.rgb2gray(image_gastro))

seg1 = seg_his(image_gastro, 'seg.png')

# lecture flux vidéo
cap = cv.VideoCapture('gastroscopy.mpg')
count = 0
while cap.isOpened():
    retr, frame = cap.read()
    if count == 0:  # récupère les dimensions au début
        dimensions = frame.shape
        centrex, centrey = dimensions[1] / 2, dimensions[0] / 2
        dim = (int(centrex), int(centrey))
    # cv.imshow('window-name', frame)
    # cv.imwrite("frames/frame%d.png" % count, frame)
    # segmentation
    frame2gray = skimage.util.img_as_ubyte(skimage.color.rgb2gray(frame))
    ret = seg_his(frame2gray, "frames/frame%d.png" % count)
    # score
    sco= score(ret, dim)
    # resize pour affichage propre
    ret = cv.resize(ret, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
    frame = cv.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
    # concatene les deux images pour comparaison
    numpy_h_concat = np.hstack((frame, skimage.color.gray2rgb(ret)))
    # add score + n frame à l'image
    image = cv.putText(numpy_h_concat, 'Score = %d' % sco, (5, 400), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                       cv.LINE_AA)
    image = cv.putText(image, 'Frame %d' % count, (5, 370), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                       cv.LINE_AA)
    # show dans la fenêtre
    cv.imshow('comparison', image)
    if cv.waitKey(5) & 0xFF == ord('p'):
        while True:
            if cv.waitKey(5) & 0xFF == ord('s'):
                break
    count = count + 1
cap.release()
cv.destroyAllWindows()
