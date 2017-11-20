# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:05:24 2017

@author: Tze
"""

# -*- coding: utf-8 -*-
#
#from tensorflow.examples.tutorials.mnist import input_data
#
#mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.resize_images(image_decoded, [280, 280])
  return image_resized, label

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

#imagelist = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...]) sample code, files for respective images
trainLabels = tf.constant(coordTrainLabels)
print("Length of imagelist is {}".format(len(imageTrainList)))
print("Length of labels is {}".format(len(coordTrainLabels)))

trainImages = tf.constant(imageTrainList)
#labels = tf.constant([0, 37, ...]) - sample code, labels for respective images
datasetTrain = tf.contrib.data.Dataset.from_tensor_slices((trainImages, trainLabels))
datasetTrain = datasetTrain.map(_parse_function)
datasetTrain = datasetTrain.repeat(-1)

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

#imagelist = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...]) sample code, files for respective images
testLabels = tf.constant(coordTestLabels)
print("Length of testList is {}".format(len(imageTestList)))
print("Length of testLabels is {}".format(len(coordTestLabels)))

testImages = tf.constant(imageTestList)
#labels = tf.constant([0, 37, ...]) - sample code, labels for respective images
datasetTest = tf.contrib.data.Dataset.from_tensor_slices((testImages, testLabels))
datasetTest = datasetTest.map(_parse_function)
datasetTest = datasetTest.repeat(-1)


#with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        
##        inc_dataset = tf.contrib.data.Dataset.range(100)
##        dec_dataset = tf.contrib.data.Dataset.range(0, -100, -1)
##        dataset = tf.contrib.data.Dataset.zip((inc_dataset, dec_dataset))
#        batched_dataset = dataset.batch(50)
#        iterator = batched_dataset.make_one_shot_iterator()
#        next_element = iterator.get_next()

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
y_= tf.placeholder(tf.float32,[None,2])

x_image=tf.reshape(x,[-1,280,280,1])
##############
h_pool0 = max_pool_2x2(x_image)
h_pool5 = max_pool_2x2(h_pool0)

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(h_pool5,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([18*18*64,1024])
b_fc1= bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool2,[-1,18*18*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
######

#W_conv1 = weight_variable([5,5,1,32])
#b_conv1 = bias_variable([32])
#h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)
#
#W_conv2 = weight_variable([5,5,32,64])
#b_conv2 = bias_variable([64])
#h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)
#
#W_fc1 = weight_variable([35*35*64,1024])
#b_fc1= bias_variable([1024])
#
#h_pool3_flat = tf.reshape(h_pool2,[-1,35*35*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)
#
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#W_fc2 = weight_variable([1024,2])
#b_fc2 = bias_variable([2])
#
#y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

######

#distance = tf.reduce_sum(tf.pow(y_conv-y_,2))
deviation = tf.reduce_mean(tf.abs(y_-y_conv))
debug = tf.abs(y_-y_conv)

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(deviation)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

start_time = time.time()

graphingData = np.zeros(30000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #Training Variables
    batched_dataset = datasetTrain.batch(50)
    iterator = batched_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)
    
    #Test Variables
    testDataset = datasetTest.batch(50)
    testIterator = testDataset.make_initializable_iterator()
    test_element = testIterator.get_next()
    sess.run(testIterator.initializer)
    for i in range(2000):
        if(i%1000==0):
            print(i)
            
        #Train Iterator
        iteration = sess.run(next_element)
        feed = {x: iteration[0], y_: iteration[1], keep_prob: 0.5}
        #Test Iterator
        testIteration = sess.run(test_element)
        testfeed = {x: testIteration[0],y_:testIteration[1],keep_prob:1.0}
        graphingData[i]=sess.run(deviation,feed_dict=testfeed)
        
        if i % 100 == 0:
#            print(graphingData)
            print("Pixel Deviation: {}".format(sess.run(deviation,feed_dict=feed)))
        sess.run(train_step,feed_dict=feed)
#        print(sess.run(y_conv,feed_dict=feed))
#        print(sess.run(deviation,feed_dict=feed))
    
#    testDataset = datasetTrain.batch(1)
#    testIterator = testDataset.make_initializable_iterator()
#    test_element = testIterator.get_next()
#    sess.run(testIterator.initializer)
#    print("These are the Test Phase results")
#    for j in range(50):
#        testIteration = sess.run(test_element)
#        testfeed = {x: testIteration[0],y_:testIteration[1],keep_prob:1.0}
##        sess.run(train_step,feed_dict = testfeed)
#        print("Image coordinate {}".format(sess.run(y_,feed_dict=testfeed)))
#        print("Predicted coordinate {}".format(sess.run(y_conv,feed_dict=testfeed)))
    
#        print(sess.run(y_conv,feed_dict=feed))
#        print(sess.run(y_,feed_dict=feed))
#        print(sess.run(debug,feed_dict=feed))

#            print('test accuracy %g' % accuracy.eval(feed_dict={
#                    x: batched_dataset.images, y_: batched_dataset, keep_prob: 1.0}))

print(time.time()-start_time)

plt.plot(graphingData[10:2000])
plt.ylabel('Average Pixel Distance')
plt.xlabel('Iteration Number')
plt.title('Pixel distance vs Iteration')
plt.show()