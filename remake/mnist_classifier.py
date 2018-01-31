import tensorflow as tf
import numpy as np
import cv2
import sys

sys.path.append("utils/")
from mnist_classification_without_null_variables import get_mnist_classification_variables

def classify_images(list_of_crops: np.ndarray):
    model_path = "./models/mnist_fc_without_null/model.ckpt"
    tf.reset_default_graph()

    v = get_mnist_classification_variables()
    x, y, keep_prob = v["x"], v["y"], v["keep_prob"]
    y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    results = []
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        for crops in list_of_crops:
            labels = np.zeros((len(crops), 11))
            feed_dict = {x: crops, y: labels, keep_prob: 1.0}

            res = sess.run(y_conv, feed_dict)
            """
            out = [(np.argmax(r), np.max(r)) for r in res]
            result = max(out, key=lambda t: t[1])

            results.append(result)
            """
            res_softmax = sess.run(tf.nn.softmax(res)) # softmax to get the probabilities
            sum_of_confidences = np.sum(res_softmax, axis=0) # sum up the probabilities across crops
            
            results.append((np.argmax(sum_of_confidences), np.max(sum_of_confidences)))
            

    return results

if __name__ == "__main__":
    image = cv2.imread("data/raw_mnist/0/0_0000.png", 0)
    image = np.reshape(image, (28,28,1))
    print(classify_images([[image]]))
