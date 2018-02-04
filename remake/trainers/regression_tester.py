import tensorflow as tf
import numpy as np
import cv2
import os
import sys

sys.path.append("../utils/")
from coordinate_regression_variables import get_coordinate_regression_variables

# declaring parameter values
model_path = "../models/localization_regression/model.ckpt"
data_source = "../data/localization_data/distributed/test_set"
image_height, image_width = 280, 280
batch_size = 50

# converting data to tensors
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
    return image_resized, label

def get_dataset(label):
    data_path = os.path.join(data_source, str(label))
    
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_path, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(' ')
            label, index, y, x = map(int, t)
            coord_labels.append((y, x))

    image_list = []
    for file in os.listdir(data_path):
        if file.endswith(".png"):
            image_list.append(os.path.join(data_path, file))

    assert len(coord_labels) == len(image_list), "Mismatch in number of inputs"

    labels = tf.constant(coord_labels)
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(-1)

    return dataset

v = get_coordinate_regression_variables()
x, y_, keep_prob = v["x"], v["y_"], v["keep_prob"]
y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]

init = tf.global_variables_initializer()
saver = tf.train.Saver()

accumulated_accuracy = 0.
total_iterations = 0
    
for i in range(10):
    print("Label:", i)
    dataset = get_dataset(i)
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)

        for i in range(10):
            iteration = sess.run(next_element)
            feed_dict = {x: iteration[0], y_: iteration[1], keep_prob: 1.0}
            sess.run(train_step, feed_dict=feed_dict)

            test_accuracy = accuracy.eval(feed_dict)
            print(i, accuracy.eval(feed_dict))
            accumulated_accuracy+= test_accuracy
            total_iterations+= 1

print("Total average accuracy:", accumulated_accuracy/total_iterations)
