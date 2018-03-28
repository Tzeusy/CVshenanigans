import numpy as np
import tensorflow as tf
import cv2
import random
import sys

sys.path.append('./utils')
from dataset import Dataset
from regressor import Regressor
from classifier import Classifier
from rnn import RNN

regressor_model = 'models/regressor/model.ckpt'
classifier_model = 'models/classifier/model.ckpt'
rnn_model = 'models/rnn_classifier/model.ckpt'

class Exec:
    def __init__(self):
        self.offsets = range(-3, 4)
        #self.offsets = [-6, -3, 0, 3, 6]

        self.regressor = Regressor()
        self.classifier = Classifier()

        time_steps = len(self.offsets) ** 2
        n_input = self.classifier.size ** 2
        num_classes = self.classifier.num_classes
        self.rnn = RNN(time_steps, n_input, num_classes)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.regressor.saver.restore(self.sess, regressor_model)
        self.classifier.saver.restore(self.sess, classifier_model)


    def crop(self, image, x, y):
        size = self.classifier.size
        crop = image[y - size//2 : y + size//2, x - size//2 : x + size//2]

        if crop.shape[:2] != (size, size):
            return None
        return np.reshape(crop, (size, size, self.classifier.num_channels))

    def show_results(self, images, coords):
        for image, coord in zip(images, coords):
            size = self.classifier.size
            num_channels = self.classifier.num_channels
            x, y = coord

            stitch_shape = (len(self.offsets) * size, len(self.offsets) * size, num_channels)
            stitch = np.zeros(stitch_shape, np.uint8)
            for i, dx in enumerate(self.offsets):
                for j, dy in enumerate(self.offsets):
                    crop = self.get_crop(image, x + dx, y + dy)
                    if crop is None:
                        crop = np.ones((size, size, num_channels), np.uint8) * 255

                    s_x, s_y = (i*size, j*size)
                    stitch[s_y:s_y+size, s_x:s_x+size] = crop

            # draw rectangle on original image
            p1 = (x - size//2, y - size//2)
            p2 = (x + size//2, y + size//2)

            cv2.rectangle(image, p1, p2, (255, 255, 255), 1)
            cv2.imshow('Image', image)
            cv2.imshow('Stitch', stitch)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_coordinates(self, images):
        images = [np.reshape(image, (self.regressor.height, self.regressor.width, self.regressor.num_channels)) for image in images]
        labels = [[0, 0] for i in range(len(images))]
        feed_dict = {
            self.regressor.x: images,
            self.regressor.y: labels,
            self.regressor.keep_prob: 1.0
        }

        coords = self.sess.run(self.regressor.output, feed_dict=feed_dict)
        coords = [tuple(map(int, c)) for c in coords]
        #self.show_results(images, coords)
        return coords

    def classify_images(self, list_of_crops):
        results = []
        for crops in list_of_crops:
            labels = np.zeros((len(crops), self.classifier.num_classes), np.float32)
            feed_dict = {
                self.classifier.x: crops,
                self.classifier.y: labels,
                self.classifier.keep_prob: 1.0
            }

            softmaxes = self.sess.run(self.classifier.output, feed_dict=feed_dict)
            out = [(np.argmax(p), np.max(p)) for p in softmaxes] # take the largest probability out of each softmax as the prediction
            result = max(out, key=lambda t: t[1]) # take the largest confidence prediction

            results.append(result)

        return results

    def localize_and_classify(self, images):
        coordinates = self.get_coordinates(images)

        all_crops = []
        for i, image in enumerate(images):
            x, y = coordinates[i]

            crops = []
            for dx in self.offsets:
                for dy in self.offsets:
                    crop = self.crop(image, x + dx, y + dy)

                    if not (crop is None):
                        crops.append(crop)

            if len(crops) > 0:
                all_crops.append(crops)

        return [pred for pred, prob in self.classify_images(all_crops)]

    def cnn(self, label, images):
        results = self.localize_and_classify(images)
        counts = [results.count(i) for i in range(self.classifier.num_classes)]
        print(label, counts, round(counts[label] / sum(counts), 5))

    def rnn_train(self, images, labels, coords):
        rnn_x, rnn_y = [], []

        for image, label, coord in zip(images, labels, coords):
            x, y = coord

            crops = []
            for dx in self.offsets:
                for dy in self.offsets:
                    crop = self.crop(image, x + dx, y + dy)

                    if not (crop is None):
                        crop = np.reshape(crop, (self.classifier.size**2))
                        crops.append(crop)

            if len(crops) == self.rnn.time_steps:
                rnn_x.append(crops)
                rnn_y.append(label)

        num_iterations = 20000
        batch_size = 50
        test_size = 500

        for i in range(num_iterations):
            data = random.choices(list(zip(rnn_x, rnn_y)), k=batch_size+test_size)
            x, labels = zip(*data)

            y = np.zeros((len(labels), self.rnn.num_classes), np.float32)
            for j, label in enumerate(labels):
                y[j][label] = 1.

            feed_dict = {
                self.rnn.x: x[:batch_size],
                self.rnn.y: y[:batch_size]
            }
            self.sess.run(self.rnn.train_step, feed_dict=feed_dict)

            if i % 100 == 0:
                with self.sess.as_default():
                    accuracy = self.rnn.accuracy.eval(feed_dict={
                        self.rnn.x: x[batch_size:],
                        self.rnn.y: y[batch_size:]
                    })
                print(i, accuracy)

        print(self.rnn.saver.save(self.sess, rnn_model))
