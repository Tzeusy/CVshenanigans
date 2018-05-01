import tensorflow as tf
import dataset
from tf_utils import *


class Classifier:
    def __init__(self, size, n_channels, n_classes):
        _scope = 'classifier'

        self.x = tf.placeholder(tf.float32, shape=[None, size, size, n_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        x_image = tf.map_fn(tf.image.per_image_standardization, self.x)

        with tf.variable_scope(_scope) as scope:
            self.output = self.network(x_image, size, n_channels, n_classes)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        _vars = [v for v in tf.global_variables() if v.name.startswith(_scope)]
        self.saver = tf.train.Saver(_vars)

    def network(self, x, size, channels_0, n_classes):
        channels_1 = 32
        channels_2 = 64
        neurons_1 = 1024

        # 2 layers: convolution + max pooling
        h_pool1 = conv2d_layer(x, channels_0, channels_1)
        h_pool2 = conv2d_layer(h_pool1, channels_1, channels_2)

        # fully connected layer
        new_size = size >> 2 # 2 layers of 2x2 pooling
        fc_length = new_size * new_size * channels_2
        h_fc1 = fc_layer(h_pool2, fc_length, neurons_1)

        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([neurons_1, n_classes])
        b_fc2 = bias_variable([n_classes])

        # softmax
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_softmax = tf.nn.softmax(y_conv)

        return y_softmax

