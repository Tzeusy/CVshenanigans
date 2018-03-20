import sys
sys.path.append("../variables")
sys.path.append("../data")

import tensorflow as tf
from dataset import get_coordinate_dataset
from coordinate_regressor_variables import *
train_data_src = "../../data/localization/train"
model_path = "../../models/coordinate_regressor/model.ckpt"
num_iterations = 20000
batch_size = 50

with tf.Session(graph=coordinate_regressor_graph) as sess:
    x, y, keep_prob, y_conv, train_step, accuracy = coordinate_regressor_placeholders
    sess.run(coordinate_regressor_init)

    dataset = get_coordinate_dataset(train_data_src)    
    batched_dataset = dataset.batch(batch_size)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)
    
    for i in range(num_iterations):
        batch = sess.run(next_element)
        feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5}
        train_step.run(feed_dict=feed_dict)

        if i % 100 == 0: print(i, accuracy.eval(feed_dict=feed_dict))

    coordinate_regressor_saver.save(sess, model_path)
