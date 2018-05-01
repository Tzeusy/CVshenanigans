import tensorflow as tf
from tf_utils import *

class Regressor:
    def __init__(self, height, width, n_channels):
        _scope = 'regressor'

        self.x = tf.placeholder(tf.float32, shape=[None, width, height, n_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32)

        self.x_image = tf.map_fn(tf.image.per_image_standardization, self.x)

        with tf.variable_scope(_scope) as scope:
            self.output = self.network(self.x_image, width, height, n_channels)

        diff = tf.abs(self.output - self.y)
        loss = tf.reduce_mean(diff)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        self.accuracy = tf.reduce_mean(tf.cast(diff, tf.float32), axis=0)

        _vars = [v for v in tf.global_variables() if v.name.startswith(_scope)]
        self.saver = tf.train.Saver(_vars)

    def network(self, x, width, height, channels_0):
        channels_1 = 32
        channels_2 = 64
        neurons_1 = 1024

        # 2 layers: convolution + max pooling
        h_pool0 = max_pool_2x2(max_pool_2x2(x)) # initial 4x4 max pooling
        h_pool1 = conv2d_layer(h_pool0, channels_0, channels_1)
        h_pool2 = conv2d_layer(h_pool1, channels_1, channels_2)

        # fully connected layer
        new_width, new_height = round(width/2**4), round(height/2**4)
        fc_length = new_width * new_height * channels_2
        h_fc1 = fc_layer(h_pool2, fc_length, neurons_1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([neurons_1, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv
