import tensorflow as tf
import numpy as np
import os
import time
import sys
#import beep

sys.path.append("../utils/")
from coordinate_regression_variables import get_coordinate_regression_variables

# declaring parameter values
model_path = "../models/localization_regression_consolidated/model.ckpt"
train_data_source = "../data/localization_data/consolidated/training_set"
test_data_source = "../data/localization_data/consolidated/test_set"
image_height, image_width = 280, 280
batch_size = 50
num_iterations = 30000

def get_dataset(data_source):
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_source, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(' ')
            label, index, y, x = map(int, t)
            coord_labels.append((y, x))

    image_list = []
    for file in os.listdir(data_source):
        if file.endswith(".png"):
            image_list.append(os.path.join(data_source, file))

    assert len(coord_labels) == len(image_list), "Mismatch in number of inputs"

    # converting data to tensors
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
        return image_resized, label

    labels = tf.constant(coord_labels)
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(-1)

    return dataset

v = get_coordinate_regression_variables()
x, y_, keep_prob = v["x"], v["y_"], v["keep_prob"]
y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]

init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_dataset = get_dataset(train_data_source)
test_dataset = get_dataset(test_data_source)

with tf.Session() as sess:
    start_time = time.time()
    sess.run(init)

    train_batched_dataset = train_dataset.batch(batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()

    test_batched_dataset = test_dataset.batch(500)
    test_iterator = test_batched_dataset.make_initializable_iterator()
    test_next_element = test_iterator.get_next()

    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)

    test_iteration = sess.run(test_next_element)
    test_feed_dict = {x: test_iteration[0], y_:test_iteration[1], keep_prob:1}

    for i in range(num_iterations):
        train_iteration = sess.run(train_next_element)
        train_feed_dict = {x: train_iteration[0], y_: train_iteration[1], keep_prob: 0.5}
        sess.run(train_step, feed_dict=train_feed_dict)

        if (i+1) % 100 == 0:
            print(i+1, accuracy.eval(test_feed_dict))

        if (i+1) % 1000 == 0:
            test_iteration = sess.run(test_next_element)
            test_feed_dict = {x: test_iteration[0], y_:test_iteration[1], keep_prob:1}

    file_path = saver.save(sess, model_path)
    print("Model saved in file:", file_path)
    print("Time elapsed: ", time.time()-start_time)
