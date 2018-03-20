import tensorflow as tf
from tf_variable_functions import *

width, height = 280, 280
num_channels = 1
num_output_channels_1 = 32
num_output_channels_2 = 64
num_neurons_1 = 1024
num_dimensions = 2

coordinate_regressor_graph = tf.Graph()
with coordinate_regressor_graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, width, height, num_channels])
    y = tf.placeholder(tf.float32, shape=[None, num_dimensions])

    x_image = tf.reshape(x, [-1, width, height, num_channels])
    x_image = tf.map_fn(tf.image.per_image_standardization, x_image)

    # double max pooling to reduce image size
    h_pool0 = max_pool_2x2(max_pool_2x2(x_image)) # 70x70

    # first layer
    W_conv1 = weight_variable([5,5,num_channels,num_output_channels_1])
    b_conv1 = bias_variable([num_output_channels_1])

    h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # 35x35

    # second layer
    W_conv2 = weight_variable([5,5,num_output_channels_1,num_output_channels_2])
    b_conv2 = bias_variable([num_output_channels_2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # 17x17

    # fully-connected layer
    new_width, new_height = round(width/2**4), round(height/2**4)
    flattened_size = new_width * new_height * num_output_channels_2
    W_fc1 = weight_variable([flattened_size, num_neurons_1])
    b_fc1 = bias_variable([num_neurons_1])

    h_pool2_flat = tf.reshape(h_pool2, [-1, flattened_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # last step
    W_fc2 = weight_variable([num_neurons_1, num_dimensions])
    b_fc2 = bias_variable([num_dimensions])

    # result
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # loss function
    diff = tf.abs(y - y_conv)
    deviation = tf.reduce_mean(diff)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(deviation)

    accuracy = tf.reduce_mean(tf.cast(diff, tf.float32), axis=0)
    
    coordinate_regressor_init = tf.global_variables_initializer()
    coordinate_regressor_saver = tf.train.Saver()
    coordinate_regressor_placeholders = (x, y, keep_prob, y_conv, train_step, accuracy)
