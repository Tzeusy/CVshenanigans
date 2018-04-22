import numpy as np
import tensorflow as tf
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
        self.offset = 12

        self.regressor = Regressor()
        self.classifier = Classifier()

        time_steps = self.classifier.size + (2 * self.offset)
        n_input = time_steps
        num_classes = self.classifier.num_classes
        self.rnn = RNN(time_steps, n_input, num_classes)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.regressor.saver.restore(self.sess, regressor_model)
        self.classifier.saver.restore(self.sess, classifier_model)

    def localize(self, images):
        images = [np.reshape(image, (self.regressor.height, self.regressor.width, self.regressor.num_channels)) for image in images]
        labels = [[0, 0] for _ in range(len(images))]
        feed_dict = {
            self.regressor.x: images,
            self.regressor.y: labels,
            self.regressor.keep_prob: 1.0
        }

        coords = self.sess.run(self.regressor.output, feed_dict=feed_dict)
        coords = [tuple(map(int, c)) for c in coords]
        return coords

    def classify(self, crops):
        labels = np.zeros((len(crops), self.classifier.num_classes), np.float32)

        feed_dict = {
            self.rnn.x: crops,
            self.rnn.y: labels
        }

        self.rnn.saver.restore(self.sess, rnn_model)
        results = self.sess.run(self.rnn.softmax, feed_dict=feed_dict)
        return [(np.argmax(a), np.max(a)) for a in results]

    def localize_and_classify(self, images):
        offset = self.offset + self.classifier.size // 2
        coordinates = self.localize(images)

        crops = []
        for i, image in enumerate(images):
            x, y = coordinates[i]

            x = max(x, offset)
            x = min(x, self.regressor.width - offset)
            y = max(y, offset)
            y = min(y, self.regressor.height - offset)

            crop = image[y-offset: y+offset, x-offset: x+offset]
            crops.append(crop)

        return [pred for pred, prob in self.classify(crops)]

    def rnn_test(self, label, images):
        results = self.localize_and_classify(images)
        counts = [results.count(i) for i in range(self.classifier.num_classes)]
        print(label, counts, round(counts[label] / sum(counts), 5))

    def rnn_train(self, images, labels, coords):
        offset = self.offset + self.classifier.size // 2
        rnn_x, rnn_y = [], []

        coords = []
        for i in range(60000//500):
            coords.extend(self.localize(images[i*500:(i+1)*500]))

        assert len(images) == len(labels) == len(coords)

        for image, label, coord in zip(images, labels, coords):
            x, y = coord

            x = max(x, offset)
            x = min(x, self.regressor.width - offset)
            y = max(y, offset)
            y = min(y, self.regressor.height - offset)

            crop = image[y-offset: y+offset, x-offset: x+offset]
            rnn_x.append(crop)
            rnn_y.append(label)

        num_iterations = 100000
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

        print('Model saved:', self.rnn.saver.save(self.sess, rnn_model))
