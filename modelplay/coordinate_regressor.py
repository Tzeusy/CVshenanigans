import tensorflow as tf
import numpy as np
import cv2
import os
import sys
sys.path.append("utils/variables")

from coordinate_regressor_variables import *
coordinate_regressor_model_path = "models/coordinate_regressor/model.ckpt"
coordinate_regressor_session = tf.Session(graph=coordinate_regressor_graph)
coordinate_regressor_session.run(coordinate_regressor_init)
coordinate_regressor_saver.restore(coordinate_regressor_session, coordinate_regressor_model_path)
default_session = tf.Session()

def get_coordinates(images):
    x, y, keep_prob, y_conv, train_step, accuracy = coordinate_regressor_placeholders

    images = [np.reshape(image, (height, width, num_channels)) for image in images]
    labels = [[0]*num_dimensions for i in range(len(images))]
    feed_dict = {x: images, y: labels, keep_prob: 1.0}
    
    result = coordinate_regressor_session.run(y_conv, feed_dict)
    return [tuple(map(int, t)) for t in result]

if __name__ == "__main__":
    from pathlib import Path
    filename = str(Path("data", "localization", "test", "0_0000.png"))
    image = cv2.imread(filename, 0)
    print(get_coordinates([image, image])) # 176, 65
