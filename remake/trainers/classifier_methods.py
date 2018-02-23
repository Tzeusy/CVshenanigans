# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
import os
import sys
import time
import numpy as np
import cv2
import random

from tensorflow.contrib.data import Dataset, Iterator
from imageVisualizer import show_small_image
sys.path.append("../utils/")
from mnist_classification_with_null_variables import get_mnist_classification_variables
from dataset_creation import parse_function

# classifier_model_path = "../models/mnist_fc/model.ckpt"
# train_classifier_data_source = "../data/raw_mnist/jointTraining"
# test_classifier_data_source = "../data/raw_mnist/jointTest"
#
# # IMPORTING NEURAL NETWORK STRUCTURE
# c = get_mnist_classification_variables()
# x, y, keep_prob = c["x"], c["y"], c["keep_prob"]
# y_conv, train_step, accuracy = c["y_conv"], c["train_step"], c["accuracy"]
# # IMPORTING NEURAL NETWORK STRUCTURE

def classifyNumpyArr(image):
    ##remove this line later
    tf.reset_default_graph()
    classifier_model_path = "../models/mnist_fc/model.ckpt"

    # IMPORTING NEURAL NETWORK STRUCTURE
    c = get_mnist_classification_variables()
    classifier_x, classifier_y, classifier_keep_prob = c["x"], c["y"], c["keep_prob"]
    classifier_y_conv, classifier_train_step, classifier_accuracy = c["y_conv"], c["train_step"], c["accuracy"]
    # IMPORTING NEURAL NETWORK STRUCTURE
    saver = tf.train.Saver()
    image = tf.convert_to_tensor(image)
    #image is now a [28,28] tensor
    #we now have to expand it to a [1,28,28,1] tensor for single-image analysis
    image = tf.expand_dims(image,0)
    image = tf.expand_dims(image,-1)
    label = tf.constant(np.zeros(11))
    label = tf.expand_dims(label,0)
    dataset = tf.data.Dataset.from_tensor_slices(([image],[label]))
    iterator = dataset.make_initializable_iterator()
    nexter = iterator.get_next()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, classifier_model_path)
        sess.run(iterator.initializer)
        testIteration = sess.run(nexter)
        feedDictionary = {classifier_x:testIteration[0],classifier_y:testIteration[1],classifier_keep_prob:1}
        prediction = classifier_y_conv.eval(feedDictionary)
        value = sess.run(tf.argmax(prediction[0]))
        print("Predicted value is {}".format(value))
        print("Scores: {}".format(prediction[0]))
        show_small_image(testIteration)
        return(value,prediction[0])

def classifyImagePath(imagePath):
    classifier_model_path = "../models/mnist_fc/model.ckpt"
    saver = tf.train.Saver()
    label = tf.constant(np.zeros(11))
    image, label, filename = parse_function(imagePath,label)
    image = tf.expand_dims(image,0)
    label = tf.expand_dims(label,0)
    dataset = tf.data.Dataset.from_tensor_slices(([image],[label]))
    iterator = dataset.make_initializable_iterator()
    nexter = iterator.get_next()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, classifier_model_path)
        sess.run(iterator.initializer)
        testIteration = sess.run(nexter)
        feedDictionary = {x:testIteration[0],y:testIteration[1],keep_prob:1}
        prediction = y_conv.eval(feedDictionary)
        value = sess.run(tf.argmax(prediction[0]))
        print("Predicted value is {}".format(value))
        print("Scores: {}".format(prediction[0]))
        show_small_image(testIteration)
        return(value,prediction[0])

if __name__ == "__main__":
    train_classifier_data_source = "../data/raw_mnist/jointTraining"
    test_classifier_data_source = "../data/raw_mnist/jointTest"
    num_iterations = 30000
    imageCount = 0;
    imageNumpyArray = []
    directoryImages = os.listdir(test_classifier_data_source)
    scrambledImages = sorted(directoryImages,key=lambda x:random.random())
    #Grab 10 random images from the test folder
    for images in scrambledImages:
        imagePath = os.path.join(test_classifier_data_source,images)
        imageNumpyArray.append(cv2.imread(imagePath,flags=0))
        imageCount+=1
        if(imageCount>10):
            break
    #Classify the 10 images
    for images in imageNumpyArray:
        classifyNumpyArr(images)

    # # Classify the 10 image paths
    # anotherCount=0
    # for images in scrambledImages:
    #     path = os.path.join(test_classifier_data_source,images)
    #     classifyImagePath(path)
    #     anotherCount+=1
    #     if(anotherCount>10):
    #         break
