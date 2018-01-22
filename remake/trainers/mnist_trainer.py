# https://www.tensorflow.org/get_started/mnist/pros
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf

model_path = "../models/mnist_fc/model.ckpt"

# x is the images, while y is the result
x = tf.placeholder(tf.float32, shape=[None, 784])
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

# differences from beginner
#   1. replace gradient descent with ADAM
#   2. include keep_prob in the parameters of feed_dict to control dropout rate
#   3. add logging to every 100th iteration in the process
#   4. use tf.Session instead of tf.InteractiveSession - separates the process of creating the graph from evaluating the graph

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # log total accuracy
    accumulated_accuracy = 0
    iterations = 0
    
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5}
        
        train_step.run(feed_dict=feed_dict)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("step %d, training accuracy %g" % (i, train_accuracy))

            accumulated_accuracy+= train_accuracy
            iterations+= 1

    print("Final accuracy:", accumulated_accuracy/iterations)
    saver.save(sess, model_path)
 
