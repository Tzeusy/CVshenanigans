import tensorflow as tf
from dataset import Dataset

import tensorflow as tf


class CNN:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool(self, x): # 2x2 max pooling
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def conv_layer(self, inp, input_channels, output_channels):
        W = self.weight_variable([5, 5, input_channels, output_channels])
        b = self.bias_variable([output_channels])

        h = tf.nn.relu(self.conv2d(inp, W) + b)
        h_pool = self.max_pool(h)

        return h_pool

    def fc_layer(self, inp, length, num_neurons):
        W = self.weight_variable([length, num_neurons])
        b = self.bias_variable([num_neurons])

        h_flat = tf.reshape(inp, [-1, length])
        h_fc = tf.nn.relu(tf.matmul(h_flat, W) + b)

        return h_fc

class Rcnn(CNN):
    scope = 'rcnn'
    model = '../models/rcnn/model.ckpt'
    train_data = '../data/localization/train'
    test_data = '../data/localization/test'

    def __init__(self, width=280, height=280, num_channels=1):
        self.width = width
        self.height = height
        self.num_channels = num_channels

        self.x = tf.placeholder(tf.float32, shape=[None, width, height, num_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        self.x_image = tf.map_fn(tf.image.per_image_standardization, self.x)

        with tf.variable_scope(Rcnn.scope) as scope:
            self.output = self.network(self.x_image, width, height, num_channels)

        diff = tf.abs(self.output - self.y)
        loss = tf.reduce_mean(diff)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        self.accuracy = tf.reduce_mean(tf.cast(diff, tf.float32), axis=0)

        vars = [v for v in tf.global_variables() if v.name.startswith(Rcnn.scope)]
        self.saver = tf.train.Saver(vars)

    def network(self, x, width, height, initial_channels):
        channels_1 = 32
        channels_2 = 64
        neurons_1 = 1024

        # 2 layers: convolution + max pooling
        h_pool0 = self.max_pool(self.max_pool(x)) # initial 4x4 max pooling
        h_pool1 = self.conv_layer(h_pool0, initial_channels, channels_1)
        h_pool2 = self.conv_layer(h_pool1, channels_1, channels_2)

        

        # fully connected layer
        new_width, new_height = round(width/2**4), round(height/2**4)
        fc_length = new_width * new_height * channels_2
        h_fc1 = self.fc_layer(h_pool2, fc_length, neurons_1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([neurons_1, 2])
        b_fc2 = self.bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_conv


def train():
    rcnn = Rcnn()
    num_iterations = 20000

    dataset = Dataset.localization(Rcnn.train_data)
    iterator = Dataset.iterator(dataset, batch_size=50)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        for i in range(num_iterations):
            x, y = sess.run(next_element)
            feed_dict= {
                rcnn.x: x,
                rcnn.y: y,
                rcnn.keep_prob: 0.5
            }
            rcnn.train_step.run(feed_dict=feed_dict)

            if (i+1) % 100 == 0:
                accuracy = rcnn.accuracy.eval(feed_dict=feed_dict)
                print(i+1, accuracy)

        print(saver.save(sess, rcnn.model))


if __name__ == '__main__':
    train()