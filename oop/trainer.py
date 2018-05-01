import sys
sys.path.insert(0, './utils')
import random
import numpy as np
import tensorflow as tf
import params
import data_loader
import model_loader
import inference
import dataset
from regressor import Regressor
from classifier import Classifier
from rnn import RNN

def train_regressor(num_iterations=20000):
    p = params.regressor
    regressor = Regressor(p.height, p.width, p.n_channels)

    data = dataset.localization(p.train_data)
    iterator = dataset.iterator(data)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        for i in range(num_iterations):
            x, y = sess.run(next_element)
            feed_dict = {
                regressor.x: x,
                regressor.y: y,
                regressor.keep_prob: 0.5,
            }
            regressor.train_step.run(feed_dict=feed_dict)

            if (i+1) % 100 == 0:
                accuracy = regressor.accuracy.eval(feed_dict=feed_dict)
                print(i+1, accuracy)

        print('Training complete')
        print('Model saved: ', saver.save(sess, p.model_path))


def train_classifier(num_iterations=20000):
    p = params.classifier
    classifier = Classifier(p.size, p.n_channels, p.n_classes)

    data = dataset.mnist(p.train_data)
    iterator = dataset.iterator(data)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        for i in range(num_iterations):
            x, y = sess.run(next_element)
            feed_dict = {
                classifier.x: x,
                classifier.y: y,
                classifier.keep_prob: 0.5,
            }
            classifier.train_step.run(feed_dict=feed_dict)

            if (i+1) % 100 == 0:
                accuracy = classifier.accuracy.eval(feed_dict=feed_dict)
                print(i+1, accuracy)

        print('Training complete')
        print('Model saved', saver.save(sess, p.model_path))


def test_classifier():
    p = params.classifier
    classifier = Classifier(p.size, p.n_channels, p.n_classes)

    batch_size = 1000
    num_iterations = 10000 // batch_size

    data = dataset.mnist(p.test_data)
    iterator = dataset.iterator(data, batch_size=batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, p.model_path)

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        accuracies = []
        for i in range(num_iterations):
            x, y = sess.run(next_element)
            accuracy = classifier.accuracy.eval(feed_dict={
                classifier.x: x,
                classifier.y: y,
                classifier.keep_prob: 1.0
            })
            accuracies.append(accuracy)

    print('MNIST test set evaluated:', sum(accuracies) / len(accuracies))


def train_rnn(num_iterations=100000):
    # create a new session, with RNN and Regressor objects
    sess = tf.Session()

    time_steps = params.classifier.size + 2 * params.offset
    n_input = time_steps
    rnn = RNN(time_steps, n_input, params.rnn.n_classes)
    sess.run(tf.global_variables_initializer())

    regressor = model_loader.load_regressor(sess)

    # load images, labels, and create coordinates with regressor predictions
    images, labels, _ = zip(*data_loader.load_localization('train'))  # coords not used
    coords = []
    for i in range(60000//500):
        coords.extend(inference.localize(sess, regressor, images[i*500:(i+1)*500]))

    assert len(images) == len(labels) == len(coords)

    # grab crops for training
    offset = params.offset + params.classifier.size // 2
    rnn_x, rnn_y = [], []
    for image, label, coord in zip(images, labels, coords):
        x, y = coord

        x = max(x, offset)
        x = min(x, params.regressor.width - offset)
        y = max(y, offset)
        y = min(y, params.regressor.height - offset)

        crop = image[y - offset: y + offset, x - offset: x + offset]
        rnn_x.append(crop)
        rnn_y.append(label)

    # training parameters
    batch_size = 50
    test_size = 500

    data = list(zip(rnn_x, rnn_y))
    for i in range(num_iterations):
        batch = random.choices(data, k=batch_size + test_size)
        x, labels = zip(*batch)

        y = np.zeros((len(labels), params.rnn.n_classes), np.float32)
        for j, label in enumerate(labels):
            y[j][label] = 1.

        feed_dict = {
            rnn.x: x[:batch_size],
            rnn.y: y[:batch_size]
        }
        sess.run(rnn.train_step, feed_dict=feed_dict)

        if (i+1) % 100 == 0:
            with sess.as_default():
                accuracy = rnn.accuracy.eval(feed_dict={
                    rnn.x: x[batch_size:],
                    rnn.y: y[batch_size:]
                })
                print(i+1, accuracy)

    print('Training complete')
    print('Model saved:', rnn.saver.save(sess, params.rnn.model_path))
