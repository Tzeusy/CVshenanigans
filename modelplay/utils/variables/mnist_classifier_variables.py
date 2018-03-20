import tensorflow as tf
from tf_variable_functions import *

size = 28
num_channels = 1
num_output_channels_1 = 32
num_output_channels_2 = 64
num_neurons_1 = 1024
num_classes = 10

mnist_classifier_graph = tf.Graph()
with mnist_classifier_graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, size, size, num_channels])
    y = tf.placeholder(tf.float32, shape=[None, num_classes])

    x_image = tf.reshape(x, [-1, size, size, num_channels])
    x_image = tf.map_fn(tf.image.per_image_standardization, x_image)
    
    # first layer
    W_conv1 = weight_variable([5,5,num_channels,num_output_channels_1])
    b_conv1 = bias_variable([num_output_channels_1])
                             
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 14x14

    # second layer
    W_conv2 = weight_variable([5,5,num_output_channels_1,num_output_channels_2])
    b_conv2 = bias_variable([num_output_channels_2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 7x7

    # fully-connected layer
    new_size = size >> 2
    flattened_size = new_size * new_size * num_output_channels_2
    W_fc1 = weight_variable([flattened_size, num_neurons_1])
    b_fc1 = bias_variable([num_neurons_1])

    h_pool2_flat = tf.reshape(h_pool2, [-1, flattened_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([num_neurons_1, num_classes])
    b_fc2 = bias_variable([num_classes])

    # result
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_softmax = tf.nn.softmax(y_conv)

    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize variables
    mnist_classifier_init = tf.global_variables_initializer()
    mnist_classifier_saver = tf.train.Saver()
    mnist_classifier_placeholders = (x, y, keep_prob, y_softmax, train_step, accuracy)
