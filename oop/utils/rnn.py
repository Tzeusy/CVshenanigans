import tensorflow as tf
from tensorflow.contrib import rnn


class RNN:
    scope = 'rnn'
    num_units = 128

    def __init__(self, time_steps=49, n_input=784, num_classes=10):
        self.time_steps = time_steps
        self.n_input = n_input
        self.num_classes = num_classes

        self.x = tf.placeholder(tf.float32, shape=[None, time_steps, n_input])
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes])

        self.x = tf.divide(self.x, 255.)

        with tf.variable_scope(RNN.scope) as scope:
            self.output = self.network(self.x, time_steps, num_classes)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        vars = [v for v in tf.global_variables() if v.name.startswith(RNN.scope)]
        self.saver = tf.train.Saver(vars)

    def network(self, x, time_steps, num_classes):
        input = tf.unstack(x, time_steps, axis=1)

        lstm_layer = rnn.BasicLSTMCell(RNN.num_units, forget_bias=1)
        outputs, _ = rnn.static_rnn(lstm_layer, input, dtype=tf.float32)

        W = tf.Variable(tf.random_normal([RNN.num_units, num_classes]))
        b = tf.Variable(tf.random_normal([num_classes]))

        prediction = tf.matmul(outputs[-1], W) + b
        return prediction