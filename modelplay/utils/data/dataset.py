import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
sys.path.append("../variables")

import cv2

from mnist_classifier_variables import size, num_classes
from coordinate_regressor_variables import height, width

def get_mnist_dataset(data_source):
    # parsing data from file
    image_list = []
    mnist_labels = []
    with open(Path(data_source, "label.txt")) as f:
        for line in f:
            name, label = line.strip().split(',')
            image_list.append(str(Path(data_source, name)))
            mnist_labels.append(int(label))

    depth = tf.constant(num_classes)
    one_hot_encoded = tf.one_hot(indices=mnist_labels, depth=depth)

    assert len(image_list) == len(mnist_labels), "Mismatch in number of inputs"

    # converting data to tensors
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, (size,size))
        return image_resized, label

    images = tf.constant(image_list)
    labels = one_hot_encoded
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(-1)

    return dataset


def get_coordinate_dataset(data_source):
    # parsing data from file
    image_list = []        
    coord_labels = []
    with open(Path(data_source, "label.txt")) as f:
        for line in f:
            name, label, x, y = line.rstrip().split(',')
            image_list.append(str(Path(data_source, name)))
            x, y = map(int, (x, y))
            coord_labels.append((x, y))

    assert len(image_list) == len(coord_labels), "Mismatch in number of inputs"

    # converting data to tensors
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, (height, width))
        return image_resized, label

    images = tf.constant(image_list)
    labels = tf.constant(coord_labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(-1)

    return dataset

