import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("utils/variables")

from mnist_classifier_variables import *
mnist_classifier_model_path = "models/mnist_classifier/model.ckpt"
mnist_classifier_session = tf.Session(graph=mnist_classifier_graph)
mnist_classifier_session.run(mnist_classifier_init)
mnist_classifier_saver.restore(mnist_classifier_session, mnist_classifier_model_path)

def classify_images(list_of_crops):
    x, y, keep_prob, y_softmax, train_step, accuracy = mnist_classifier_placeholders

    results = []
    for crops in list_of_crops:
        labels = np.zeros((len(crops), num_classes), np.float32)
        feed_dict = {x: crops, y: labels, keep_prob: 1.0}

        res = mnist_classifier_session.run(y_softmax, feed_dict)
        out = [(np.argmax(r), np.max(r)) for r in res]
        result = max(out, key=lambda t: t[1])

        results.append(result)

    return results

if __name__ == "__main__":
    image = cv2.imread("data/mnist/test/0_0000.png", 0)
    image = np.reshape(image, (size, size, num_channels))
    print(classify_images([[image]]))
