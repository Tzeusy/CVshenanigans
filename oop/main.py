import numpy as np
import tensorflow as tf
import cv2
import os
import sys
from pathlib import Path

sys.path.append('./utils')

from regressor import Regressor
from classifier import Classifier

train_data = Path('data/localization/train')
test_data = Path('data/localization/test')
regressor_model = 'models/regressor/model.ckpt'
classifier_model = 'models/classifier/model.ckpt'


def show_results(images, coords):
    for image, coord in zip(images, coords):
        size = classifier.size
        num_channels = classifier.num_channels
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


def get_coordinates(images):
    images = [np.reshape(image, (regressor.height, regressor.width, regressor.num_channels)) for image in images]
    labels = [[0, 0] for i in range(len(images))]
    feed_dict = {
        regressor.x: images,
        regressor.y: labels,
        regressor.keep_prob: 1.0
    }

    coords = sess.run(regressor.output, feed_dict=feed_dict)
    coords = [tuple(map(int, c)) for c in coords]
    #show_results(images, coords)
    return coords


def classify_images(list_of_crops):
    results = []
    for crops in list_of_crops:
        labels = np.zeros((len(crops), classifier.num_classes), np.float32)
        feed_dict = {
            classifier.x: crops,
            classifier.y: labels,
            classifier.keep_prob: 1.0
        }

        softmaxes = sess.run(classifier.output, feed_dict=feed_dict)
        out = [(np.argmax(p), np.max(p)) for p in softmaxes] # take the largest probability out of each softmax as the prediction
        result = max(out, key=lambda t: t[1]) # take the largest confidence prediction

        results.append(result)

    return results


def localize_and_classify(images):
    size = classifier.size
    num_channels = classifier.num_channels

    coordinates = get_coordinates(images)

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

    return [pred for pred, prob in classify_images(all_crops)]


if __name__ == '__main__':
    regressor = Regressor()
    classifier = Classifier()

    sess = tf.Session()
    regressor.saver.restore(sess, regressor_model)
    classifier.saver.restore(sess, classifier_model)

    files = [file for file in os.listdir(train_data) if file.endswith('.png')]
    for label in range(10):
        filenames = [str(train_data / file) for file in files if file[0] == str(label)][:800]
        images = [cv2.imread(name, 0) for name in filenames]

        results = localize_and_classify(images)
        counts = [results.count(i) for i in range(classifier.num_classes)]
        print(label, counts, round(counts[label]/sum(counts), 5))