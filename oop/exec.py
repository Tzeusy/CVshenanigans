import numpy as np
import tensorflow as tf
import cv2
import sys

sys.path.append('./utils')
from regressor import Regressor
from classifier import Classifier

regressor_model = 'models/regressor/model.ckpt'
classifier_model = 'models/classifier/model.ckpt'

class Exec:
    def __init__(self):
        self.regressor = Regressor()
        self.classifier = Classifier()

        self.sess = tf.Session()
        self.regressor.saver.restore(self.sess, regressor_model)
        self.classifier.saver.restore(self.sess, classifier_model)

    def show_results(self, images, coords):
        for image, coord in zip(images, coords):
            size = self.classifier.size
            num_channels = self.classifier.num_channels
            x, y = coord

            offsets = range(-3, 4)
            stitch_shape = (len(offsets) * size, len(offsets) * size, num_channels)
            stitch = np.zeros(stitch_shape, np.uint8)
            for i, dx in enumerate(offsets):
                for j, dy in enumerate(offsets):
                    _x, _y = x+dx, y+dy
                    crop = image[_y-size//2:_y+size//2, _x-size//2:_x+size//2]
                    if crop.shape != (size, size, num_channels): crop = np.ones((size, size, num_channels), np.uint8) * 255

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
        #show_results(regressor, images, coords)
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
        size = self.classifier.size
        num_channels = self.classifier.num_channels

        coordinates = self.get_coordinates(images)

        all_crops = []
        for i, image in enumerate(images):
            x, y = coordinates[i]
            offsets = range(-3, 4)

            crops = []
            for dx in offsets:
                for dx in offsets:
                    _x, _y = x+dx, y+dx
                    crop = image[_y-size//2:_y+size//2, _x-size//2:_x+size//2]

                    if crop.shape == (size, size):
                        crops.append(np.reshape(crop, (size, size, num_channels)))

            if len(crops) > 0: all_crops.append(crops)

        return [pred for pred, prob in self.classify_images(all_crops)]

    def cnn(self, label, images):
        results = self.localize_and_classify(images)
        counts = [results.count(i) for i in range(self.classifier.num_classes)]
        print(label, counts, round(counts[label] / sum(counts), 5))