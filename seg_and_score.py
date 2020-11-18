import numpy as np
import skimage.color
import skimage.io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import skimage.viewer
import cv2 as cv
import methods


def _entropy(img):
    p = np.array([(img == v).sum() for v in range(256)])
    p = p / p.sum()
    entr = -(p[p > 0] * np.log2(p[p > 0])).sum()
    return entr #np.mean(entropy(skimage.color.rgb2gray(img),disk(10)))


def mean_hs(list_h, list_s):
    _count = 0
    _sum_h = 0
    _sum_s = 0
    it = 0
    x, y = list_h.shape
    for i in range(x):
        for j in range(y):
            it += 1
            if it % 20 == 0:
                _count += 1
                _sum_h += list_h[i, j]
                _sum_s += list_s[i, j]
    return _sum_h / _count, _sum_s / _count


# segmentation (HSV based)
def seg_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(img)
    mean_h, mean_s = np.mean(np.ndarray.flatten(h)), np.mean(np.ndarray.flatten(s))
    if mean_s < 120:  # test
        tresh_s = mean_s * 0.9
    else:
        tresh_s = mean_s
    mask = cv.inRange(img, (20, 0, 170), (255, tresh_s, 240))
    return mask, mean_h, mean_s
    # return cv.bitwise_and(img, img, mask=mask)


# todo : improve score function with numpy library
# renvoie un score de qualité à partir de l'image binaire
def score(ima, dim):
    score = 0
    bad_pixels = cv.findNonZero(ima)
    if bad_pixels is not None:
        score = bad_pixels.shape[0] / (4 * dim[0] * dim[1])  # pourcentage de surface couverte par des bulles
    return score


# applique les transpho morphologiques à l'image
def morph_trans(ima):
    kernel = np.ones((7, 7), np.uint8)
    kernelb = np.ones((5, 5), np.uint8)
    ima = cv.morphologyEx(ima, cv.MORPH_CLOSE, kernel)  # clustering
    # ima = cv.medianBlur(ima, 3)
    ima = cv.morphologyEx(ima, cv.MORPH_OPEN, kernel)  # denoise
    ima = cv.dilate(ima, kernelb, iterations=1)
    return ima


image_gastro = skimage.io.imread('bulles.png')

# lecture flux vidéo
cap = cv.VideoCapture('gastroscopy.mpg')
count = 1
while not cap.isOpened():  # attente active en cas de lecture de flux en real-time, on attend le header
    cap = cv.VideoCapture("gastroscopy.mpg")
    cv.waitKey(1000)
    print("Wait for the header")
while cap.isOpened():
    retr, frame = cap.read()
    if count == 1:  # récupère les dimensions au début
        dimensions = frame.shape
        centrex, centrey = dimensions[1] / 2, dimensions[0] / 2
        dim = (int(centrex), int(centrey))
    # segmentation
    if retr:
        # frame = cv.medianBlur(frame, 5)
        ret, mean_h, mean_s = seg_hsv(frame)
        entr = _entropy(frame)
        if mean_s > 175 or entr < 6:
            sco = "garbage"  # temp
        else:
            ret = morph_trans(ret)
            # score
            sco = str(score(ret, dim) * 100)

        # resize pour affichage propre
        ret = skimage.color.gray2rgb(ret)
        ret = cv.resize(ret, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        frame = cv.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        # concatene les deux images pour comparaison
        numpy_h_concat = np.hstack((frame, ret))
        # add score + n frame à l'image
        image = cv.putText(numpy_h_concat, 'Frame %d' % count, (5, 370), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'score = %s' % sco, (5, 400), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'msat = %d' % mean_s, (5, 420), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                           cv.LINE_AA)
        image = cv.putText(image, 'entropy = %f' % entr, (5, 350), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1,
                           cv.LINE_AA)
        # show dans la fenêtre
        # cv.imshow('comparison', image)
        cv.imwrite('hsv_seg/test_sue%d_e.png' % count, image)
        count += 1
    else:  # si la frame n'est pas prête
        cv.waitKey(1)
    if cv.waitKey(1) & 0xFF == ord('p'):
        while True:
            if cv.waitKey(1) & 0xFF == ord('s'):
                break
cap.release()
cv.destroyAllWindows()
