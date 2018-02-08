import tensorflow as tf
import os
import sys
import time
import numpy as np
from tensorflow.contrib.data import Dataset, Iterator

image_height, image_width = 28, 28
#change to 11 for noNull
num_labels = 11

def labelToInt(x):
    try:
        return int(x)
    except:
        return(10)

def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, [image_height,image_width])
    return image_resized, label, filename

def get_classifier_dataset(data_source):
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_source, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(',')
            filename, value = map(labelToInt, t)
            coord_labels.append(value)
    depth = tf.constant(num_labels)
    one_hot_encoded = tf.one_hot(indices=coord_labels, depth=depth)

    image_list = []
    for file in os.listdir(data_source):
        if file.endswith(".png"):
            image_list.append(os.path.join(data_source, file))

    assert len(coord_labels) == len(image_list), "Mismatch in number of inputs"

    # converting data to tensors

    labels = one_hot_encoded
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=48000)
    dataset = dataset.repeat()
    return dataset
