# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
import os
import sys
import time
import numpy as np
from tensorflow.contrib.data import Dataset, Iterator
from imageVisualizer import show_small_image
sys.path.append("../utils/")
from mnist_classification_with_null_variables import get_mnist_classification_variables
from dataset_creation import get_classifier_dataset

model_path = "../models/mnist_fc/model.ckpt"
train_classifier_data_source = "../data/raw_mnist/jointTraining"
test_classifier_data_source = "../data/raw_mnist/jointTest"
num_iterations = 30000

# IMPORTING NEURAL NETWORK STRUCTURE
v = get_mnist_classification_variables()
x, y, keep_prob = v["x"], v["y"], v["keep_prob"]
y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]
# IMPORTING NEURAL NETWORK STRUCTURE

#Data input and iterator generation
testData = get_classifier_dataset(test_classifier_data_source).batch(1)
testIterator = testData.make_initializable_iterator()
nextTest = testIterator.get_next()
#Data input and iterator generation

#Loading Model
saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, model_path)
    sess.run(testIterator.initializer)
    testIteration = sess.run(nextTest)
    testDict = {x: testIteration[0],y:testIteration[1],keep_prob:1}
    for i in range(num_iterations):
        prediction = y_conv.eval(testDict)
        value = sess.run(tf.argmax(prediction[0]))
        print("Predicted value is {}".format(value))
        show_small_image(testIteration)
        testIteration = sess.run(nextTest)
        testDict = {x: testIteration[0],y:testIteration[1],keep_prob:1}
