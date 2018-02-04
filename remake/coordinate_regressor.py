import tensorflow as tf
import numpy as np
import cv2
import os
import sys

sys.path.append("utils/")
from coordinate_regression_variables import get_coordinate_regression_variables

def get_coordinates(images: np.ndarray):
    # declaring parameter values
    model_path = "./models/localization_regression_consolidated/model.ckpt"
    tf.reset_default_graph()

    image_height, image_width = 280, 280
    size = 28

    v = get_coordinate_regression_variables()
    x, y_, keep_prob = v["x"], v["y_"], v["keep_prob"]
    y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    tf_images = [image.reshape((image_height, image_width, 1)) for image in images]
    labels = [np.array([0,0]) for i in range(len(images))]

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        feed_dict = {x: tf_images, y_: labels, keep_prob: 1.0}
        result = sess.run(y_conv, feed_dict)
        return [tuple(map(int, t)) for t in result]

if __name__ == "__main__":
    data_source = "./data/localization_data/distributed/test_set"
    data_path = os.path.join(data_source, "0")
    image_filename = os.path.join(data_path, "0_0000.png")
    image = cv2.imread(image_filename, 0)
    print(get_coordinates([image, image]))
