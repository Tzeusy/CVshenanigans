# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
import os
import sys
import time
import numpy as np
from tensorflow.contrib.data import Dataset, Iterator

sys.path.append("../utils/")

model_path = "../models/mnist_fc_without_null/model.ckpt"
train_data_source = "../data/raw_mnist/jointTraining"
test_data_source = "../data/raw_mnist/jointTest"
image_height, image_width = 28, 28
batch_size = 50
num_iterations = 30000

def labelToInt(x):
    try:
        return int(x)
    except:
        return(10)

def get_dataset(data_source):
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_source, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(',')
            filename, value = map(labelToInt, t)
            coord_labels.append(value)
            
    depth = tf.constant(10)
    one_hot_encoded = tf.one_hot(indices=coord_labels, depth=depth)

    image_list = []
    for file in os.listdir(data_source):
        if file.endswith(".png"):
            image_list.append(os.path.join(data_source, file))

    assert len(coord_labels) == len(image_list), "Mismatch in number of inputs"

    # converting data to tensors
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
        return image_resized, label

    labels = one_hot_encoded
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=30000)
    dataset = dataset.repeat()
    
    return dataset

#v = get_mnist_classification_variables()
#x, y, keep_prob = v["x"], v["y"], v["keep_prob"]
#y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]

x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
y = tf.placeholder(tf.float32, shape=[None, 10])

# here we begin creation of a new model, involving cnn
# weights are initialized with a small amount of noise to break symmetry, and prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# since neurons are ReLU, initialize them with a slight positive bias to avoid "dead neurons"
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# implementation of the first layer
# compute 32 features for each 5x5 patch
W_conv1 = weight_variable([5,5,1,32]) # (patch_size, patch_size, # input channels, # output channels)
b_conv1 = bias_variable([32])
# reshape x to a 4d tensor, with 2nd and 3rd dimensions being width and height, and 4th dimension being number of colour channels
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x_image with W, then add b, apply the ReLU function, then max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # this reduces image size to 14x14

# implementation of the second layer
# 64 features for each 5x5
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # 7x7

# implementation of fully-connected layer with 1024 neurons - this processes based on the entire image
W_fc1 = weight_variable([7*7 * 64, 1024]) # 64 layers of 7x7 image
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) # flatten the image into a single line
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout - reduces overfitting
# dropout randomly selects units in a neural network, and temporarily removes all incoming and outgoing connections
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# layer for softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

trainData = get_dataset(train_data_source).batch(100)
testData = get_dataset(test_data_source).batch(100)

# create TensorFlow Iterator object
trainIterator = trainData.make_initializable_iterator()
nextTrain = trainIterator.get_next()

testIterator = testData.make_initializable_iterator()
nextTest = testIterator.get_next()

start_time = time.time()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(trainIterator.initializer)
    sess.run(testIterator.initializer)
    cyclingCounter=0
    movingAverage=np.zeros(10)
    for i in range(10000):
        if not i%50:
            print("Step {}".format(i))
            testIteration = sess.run(nextTest)
            testDict = {x: testIteration[0],y:testIteration[1],keep_prob:1}
            acc = sess.run(accuracy,feed_dict=testDict)
            movingAverage[cyclingCounter]=acc
            cyclingCounter+=1
            if cyclingCounter>9:
                cyclingCounter=0
            totalAverage = sum(movingAverage)/10
            print("Accuracy is {}".format(totalAverage))
        iteration = sess.run(nextTrain)
#        print(iteration[1])
        feedDictionary = {x: iteration[0], y: iteration[1], keep_prob:0.5}
        sess.run(train_step,feed_dict=feedDictionary)


print(time.time()-start_time)
