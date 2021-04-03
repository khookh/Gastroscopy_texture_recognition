import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import sys
import time


def predict(img):
    img = cv.resize(img, (160, 160))
    img = np.reshape(img, [1, 160, 160, 3])
    dnn_model.predict(img)


if __name__ == '__main__':
    start = time.time()
    dnn_model = tf.keras.models.load_model("../model.h5")  # Load trained classification model (accuracy ~ 94%)
    # model trained on kaggle with kvasir dataset https://www.kaggle.com/stefanodonne/gastroscopic-classification
    dnn_model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    end = time.time()
    print("load model time = %f" % (end - start))  # laptop = 1.448 s
    predict_time = np.array([])

    start = time.time()
    for directory in os.listdir(str(sys.argv[1])):
        dirc = os.path.join(str(sys.argv[1]), directory) + "\\"
        for file in os.listdir(dirc):
            file_path = os.path.join(dirc, file)
            predict(cv.imread(file_path))
            end = time.time()
            predict_time = np.append(predict_time, end - start)
            start = end

    print(np.mean(predict_time))  # laptop = 61.9 ms
