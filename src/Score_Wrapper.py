import numpy as np
import tensorflow as tf
import cv2 as cv


class Wrap_:
    """
    class managing the score outputing and sequencing
    """

    uniformity_list = np.array([])  # temp
    score_list = np.array([])
    temp_score_list = np.array([])  # buffer of recent score (will be discarded or added to section_score_list)
    section_score_list = np.array([])  # score of section
    dim = (0, 0)
    section = 1  # actual section number
    p_capture = False  # True if the current frames correspond to an image capture during the gastroscopic exam
    count_b_p = 0

    dnnmodel = tf.keras.models.load_model("./src/model.h5")  # Load trained classification model (accuracy ~ 94%)
    # model trained on kaggle with kvasir dataset https://www.kaggle.com/stefanodonne/gastroscopic-classification
    dnnmodel.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    class_n = ['pylorus', 'retroflex-stomach', 'z-line']  # for now only 3 classes classified

    def section_switch(self, predict):
        print("Predict class '%s' with %f perc." % (predict[0], predict[1]))
        if self.section == 1 and predict[0] == 'z-line':
            self.section += 1
            self.section_score()

    def crop(self, img):
        """
        Crop the frame for accurate DNN classification
        :param img: to be cropped
        :return: cropped image
        """
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
        """
        Predict the frame class (DNN classification) : Z-Line (oesophagus) , Retro-flex shot or Pylorus
        :param img: input frame
        :return: predicted class
        """
        img = self.crop(img)
        cv.imshow('DNN', img)  # debug
        cv.waitKey(1)  # debug
        img = cv.resize(img, (160, 160))
        img = np.reshape(img, [1, 160, 160, 3])
        prediction = self.dnnmodel.predict(img)
        index = np.argmax(prediction[0])
        print(prediction[0])
        return self.class_n[index], prediction[0][index]

    def section_score(self):
        """
        Output into the file the score of the section that has been processed
        """
        print("seq %d" % self.section)
        self.file.write("Mean score in section %i = %.2f \n" % (self.section, np.mean(self.section_score_list)))
        self.file.write("_____________________\n")
        self.score_list = np.append(self.score_list, self.section_score_list)
        self.section_score_list = np.array([])

    def save(self):
        """
        Add to score list the buffered score
        :return:
        """
        if self.temp_score_list.size > 8:
            self.section_score_list = np.append(self.section_score_list, self.temp_score_list)
        self.temp_score_list = np.array([])

    def strict_diff(self):
        """
        :return: True if the 6 previous frames are strictly different
        """
        if self.uniformity_list.size > 6:
            for i in range(6):
                if self.uniformity_list[-1 - i] == self.uniformity_list[-1 - i - 1]:
                    return False
            return True
        return False

    def strict_eq(self):
        """
        :return: True if the 4 previous frames are similar
        """
        if self.uniformity_list.size > 4:
            for i in range(4):
                if self.uniformity_list[-1 - i] != self.uniformity_list[-1 - i - 1]:
                    return False
            return True
        return False

    def w_check(self, frame):
        """
        Check if a picture is taken
        :param frame: picture taken
        :return: for now nothing (future: class predicted for section management)
        """
        self.count_b_p += 1
        if self.p_capture is False and self.strict_eq():
            self.p_capture = True
            if self.count_b_p > 100:
                self.save()
                self.section_switch(self.predict(frame))
            self.temp_score_list = np.array([])

        if self.p_capture is True and self.strict_diff():
            self.p_capture = False
            self.count_b_p = 0

    def ss_temp(self):
        """
        Save/discard buffer and add it to section score
        """
        self.save()
        self.section_score()

    def output_f(self, count):
        """
        Add meaningful info to output file and close the stream
        :param count: total frame count
        """
        self.file.write("Mean score (frame-wise) of whole video = %.2f \n" % np.mean(self.score_list))
        self.file.write("Dimension of the video (treatment) = %f x %f \n" % (self.dim[0], self.dim[1]))
        self.file.write("(%.2f %% of the frame from the video were treated)" % (self.score_list.size * 100.0 / count))
        self.file.close()

    def __init__(self, filename):
        """
        Init thread wrapper, open output file
        :param filename: name of the source file
        """
        self.fileName = filename
        self.file = open("output_%s_.txt" % self.fileName, "w")
