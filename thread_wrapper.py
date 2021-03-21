import numpy as np
import tensorflow as tf
import cv2 as cv


# class managing the score outputing and sequencing
class Wrap_:
    uniformity_list = np.array([])
    score_list = np.array([])
    temp_score_list = np.array([])
    section_score_list = np.array([])
    fps_list = np.array([])
    dim = (0, 0)
    section = 1
    sco = 0
    unfy = 0
    p_capture = False
    count_b_p = 0

    dnnmodel = tf.keras.models.load_model("./model.h5")
    # model trained on kaggle with kvasir dataset
    dnnmodel.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    class_n = ['pylorus', 'retroflex-stomach', 'z-line']

    def crop(self, img):
        size = img.shape
        h, s, v = cv.split(img)
        x1 = 1
        x2 = size[1] - 1
        y1 = 1
        y2 = size[0] - 1
        while v[int(size[0] / 2), x1] < 15:
            x1 += 5
        while v[int(size[0] / 2), x2] < 15:
            x2 -= 5
        while v[y1, int(size[1] / 2)] < 15:
            y1 += 5
        while v[y2, int(size[1] / 2)] < 15:
            y2 -= 5
        return img[y1:y2, x1:x2]

    def predict(self, img):
        img = self.crop(img)
        cv.imshow('testcrop', img) #debug
        cv.waitKey(1) #debug
        img = cv.resize(img, (160, 160))
        img = np.reshape(img, [1, 160, 160, 3])
        prediction = self.dnnmodel.predict(img)
        index = np.argmax(prediction[0])
        print(prediction[0])
        return self.class_n[index]

    # output into the file the score of the section that has been processed
    def section_score(self):
        print("seq %d" % self.section)
        self.file.write("Mean score in section %i = %.2f \n" % (self.section, np.mean(self.section_score_list)))
        self.file.write("_____________________\n")
        self.score_list = np.append(self.score_list, self.section_score_list)
        self.section_score_list = np.array([])
        self.section += 1

    # add to score list the buffered score
    def save(self):
        if self.temp_score_list.size > 8:
            self.section_score_list = np.append(self.section_score_list, self.temp_score_list)
        self.temp_score_list = np.array([])

    # Return True if the 6 previous frames are strictly different
    def strict_diff(self):
        if self.uniformity_list.size > 6:
            for i in range(6):
                if self.uniformity_list[-1 - i] == self.uniformity_list[-1 - i - 1]:
                    return False
            return True
        return False

    # Return True if the 4 previous frames are similar
    def strict_eq(self):
        if self.uniformity_list.size > 4:
            for i in range(4):
                if self.uniformity_list[-1 - i] != self.uniformity_list[-1 - i - 1]:
                    return False
            return True
        return False

    # Check if the pictures taken aren't too close apart # TODO : implement image detection
    def w_check(self, frame):
        self.count_b_p += 1
        if self.p_capture is False and self.strict_eq():
            self.p_capture = True
            if self.count_b_p > 100:
                self.save()
                print(self.predict(frame))
                # self.section_score()
            self.temp_score_list = np.array([])

        if self.p_capture is True and self.strict_diff():
            self.p_capture = False
            self.count_b_p = 0

    def ss_temp(self):
        self.save()
        self.section_score()

    def output_f(self, count):
        self.file.write("Mean score (frame-wise) of whole video = %.2f \n" % np.mean(self.score_list))
        self.file.write("Average fps = %.2f \n" % np.mean(self.fps_list))
        self.file.write("Dimension of the video (treatment) = %f x %f \n" % (self.dim[0], self.dim[1]))
        self.file.write("(%.2f %% of the frame from the video were treated)" % (self.score_list.size * 100.0 / count))
        self.file.close()

    def __init__(self, filename):
        self.fileName = filename
        self.file = open("output_%s_.txt" % self.fileName, "w")
