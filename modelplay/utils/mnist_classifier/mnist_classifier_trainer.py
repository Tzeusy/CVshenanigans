import sys
sys.path.append("../variables")
sys.path.append("../data")

import tensorflow as tf
from dataset import get_mnist_dataset
from mnist_classifier_variables import *
train_data_src = "../../data/mnist/train"
model_path = "../../models/mnist_classifier/model.ckpt"
num_iterations = 20000
batch_size = 50

with tf.Session(graph=mnist_classifier_graph) as sess:
    x, y, keep_prob, y_softmax, train_step, accuracy = mnist_classifier_placeholders
    sess.run(mnist_classifier_init)

    dataset = get_mnist_dataset(train_data_src)
    batched_dataset = dataset.batch(batch_size)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)

    for i in range(num_iterations):
        batch = sess.run(next_element)
        feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5}
        train_step.run(feed_dict)

        if i % 100 == 0: print(i)

    mnist_classifier_saver.save(sess, model_path)
