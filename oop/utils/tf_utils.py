import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x): # 2x2 max pooling
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def conv2d_layer(inp, input_channels, output_channels):
    W = weight_variable([5, 5, input_channels, output_channels])
    b = bias_variable([output_channels])

    h = tf.nn.relu(conv2d(inp, W) + b)
    h_pool = max_pool_2x2(h)

    return h_pool


def fc_layer(inp, length, num_neurons):
    W = weight_variable([length, num_neurons])
    b = bias_variable([num_neurons])

    h_flat = tf.reshape(inp, [-1, length])
    h_fc = tf.nn.relu(tf.matmul(h_flat, W) + b)

    return h_fc
