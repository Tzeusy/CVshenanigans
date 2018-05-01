import numpy as np
import params


# Given a list of images, use the regressor to determine the center of the object
#
# input - array of 280x280 images, np.ndarray: [-1, h, w, c]
# return - list of tuples of (x, y) for each input image
def localize(sess, regressor, images):
    shape = (params.regressor.height, params.regressor.width, params.regressor.n_channels)
    images = [np.reshape(image, shape) for image in images]
    labels = [(0, 0) for _ in range(len(images))]

    feed_dict = {
        regressor.x: images,
        regressor.y: labels,
        regressor.keep_prob: 1.0
    }
    coords = sess.run(regressor.output, feed_dict=feed_dict)
    coords = [tuple(map(int, c)) for c in coords]

    return coords


# Uses MNIST-trained CNN (99.2%) to classify each crop
# Returns the most confident result out of all crops
#
# input - one array of crops per image, np.ndarray: [-1, n_crops, h, w, c]
# return - list of tuples of (pred, prob) per image
def cnn_classify(sess, classifier, list_of_crops):
    results = []

    for crops in list_of_crops:
        labels = np.zeros((len(crops), params.classifier.n_classes))  # dummy labels

        feed_dict = {
            classifier.x: crops,
            classifier.y: labels,
            classifier.keep_prob: 1.0
        }
        # softmaxes contains one prediction per crop
        softmaxes = sess.run(classifier.output, feed_dict=feed_dict)

        # for each crop, take its best prediction
        out = [(np.argmax(arr), np.max(arr)) for arr in softmaxes]
        # compare predictions across all crops
        result = max(out, key=lambda tup: tup[1])
        results.append(result)

    return results


# Uses row-based RNN to classify each crop
#
# input - one crop per image, np.ndarray: [-1, h, w, c]
# return - list of tuples of (pred, prob) per image
def rnn_classify(sess, rnn, list_of_crops):
    labels = np.zeros((len(list_of_crops), params.rnn.n_classes))

    feed_dict = {
        rnn.x: list_of_crops,
        rnn.y: labels
    }
    results = sess.run(rnn.softmax, feed_dict=feed_dict)

    return [(np.argmax(arr), np.max(arr)) for arr in results]
