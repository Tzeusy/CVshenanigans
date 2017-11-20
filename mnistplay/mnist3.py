# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:20:40 2017

@author: Tze
"""

import tensorflow as tf
import os

from tensorflow.contrib.data import Dataset, Iterator

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.resize_images(image_decoded, [280, 280])
  return image_resized, label

#TRAINING SET
coordTrainLabels = []
with open('./0/label.txt') as f:
    for line in f:
        line = line.rstrip() #remove all instances of \n from each line
        a = line.split(' '); #split by spaces - first element is y, second element is x
        coordTrainLabels.append((int(a[1]),int(a[2]))) #adds a tuple of (y,x) to coord Labels
        
imageTrainList = []
for file in os.listdir('./0/'):
    if file.endswith('.png'):
        imageTrainList.append('./0/'+file)
        
#TEST SET
coordTestLabels = []
with open('./0t/label.txt') as f:
    for line in f:
        line = line.rstrip() #remove all instances of \n from each line
        a = line.split(' '); #split by spaces - first element is y, second element is x
        coordTestLabels.append((int(a[1]),int(a[2]))) #adds a tuple of (y,x) to coord Labels
        
imageTestList = []
for file in os.listdir('./0t/'):
    if file.endswith('.png'):
        imageTestList.append('./0t/'+file)

# Toy data
train_imgs = tf.constant(imageTrainList)
train_labels = tf.constant(coordTrainLabels)

val_imgs = tf.constant(imageTestList)
val_labels = tf.constant(coordTestLabels)

# create TensorFlow Dataset objects
tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
val_data = Dataset.from_tensor_slices((val_imgs, val_labels))

#Process image files to actual images
def input_parser(img_path, label):
    # convert the label to one-hot encoding
#    one_hot = tf.one_hot(label, NUM_CLASSES)

    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)

    return img_decoded, label
tr_data = tr_data.map(input_parser)
val_data = val_data.map(input_parser)

#Convert to batches of 50
tr_data = tr_data.batch(50)
val_data = val_data.batch(50)

# create TensorFlow Iterator object
iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

#ALL IMAGE PREPROCESSING IS DONE
#NOW TIME FOR NEURAL NETWORKING

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
                        
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32, [None,280,280,1])
y_ = tf.placeholder(tf.float32, [None, 2])

deviation = tf.square(y_[0]-y[0])+tf.square(y_[1]-y[1])



with tf.Session() as sess:

    # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

    # initialize the iterator on the validation data
    sess.run(validation_init_op)

    # get each element of the validation dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break