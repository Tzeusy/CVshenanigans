import sys
sys.path.append("../variables")
sys.path.append("../data")

import tensorflow as tf
from dataset import get_mnist_dataset
from mnist_classifier_variables import *
test_data_src = "../../data/mnist/test"
model_path = "../../models/mnist_classifier/model.ckpt"
batch_size = 1000
num_iterations = 10000 // batch_size

with tf.Session(graph=mnist_classifier_graph) as sess:
    x, y, keep_prob, y_softmax, train_step, accuracy = mnist_classifier_placeholders
    sess.run(mnist_classifier_init)
    mnist_classifier_saver.restore(sess, model_path)

    dataset = get_mnist_dataset(test_data_src)
    batched_dataset = dataset.batch(batch_size)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)

    accuracies = []
    for i in range(num_iterations):
        batch = sess.run(next_element)
        feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0}
        accuracies.append(accuracy.eval(feed_dict=feed_dict))
        
    print(sum(accuracies)/len(accuracies))
