import tensorflow as tf
import numpy as np
import cv2
import os
import sys

sys.path.append("../utils/")
from coordinate_regression_variables import get_coordinate_regression_variables
from imageVisualizer import show_image_with_crosshairs

# declaring parameter values
model_path = "../models/localization_regression_consolidated/model.ckpt"
data_source = "../data/localization_data/consolidated/test_set"
image_height, image_width = 280, 280
# batch_size = 50


# converting data to tensors
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=1)
    image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
    return image_resized, label, filename

def get_dataset(label):
    if(label==""):
        data_path = data_source
    else: data_path = os.path.join(data_source, str(label))

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

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    dataset = get_dataset("")
    #we use 1 so humans can see
    batched_dataset = dataset.batch(1)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)
    for i in range(10000):
        # batched_dataset = dataset.batch(1)
        iteration = sess.run(next_element)
        feed_dict = {x: iteration[0], y_: iteration[1], keep_prob: 1.0}
        predicted_value = y_conv.eval(feed_dict)
        print("{},{}".format(i,predicted_value))
        show_image_with_crosshairs(iteration,predicted_value)
