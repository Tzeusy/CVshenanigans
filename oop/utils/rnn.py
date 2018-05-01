import tensorflow as tf
from tensorflow.contrib import rnn


class RNN:
    n_units = 128

    def __init__(self, time_steps, n_input, n_classes):
        _scope = 'rnn'

        x = tf.placeholder(tf.float32, shape=[None, time_steps, n_input])
        self.x = tf.divide(x, 255.)
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])

        with tf.variable_scope(_scope) as scope:
            self.output = self.network(self.x, time_steps, n_classes)
        self.softmax = tf.nn.softmax(self.output)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        _vars = [v for v in tf.global_variables() if v.name.startswith(_scope)]
        self.saver = tf.train.Saver(_vars)

    def network(self, x, time_steps, n_classes):
        input = tf.unstack(x, time_steps, axis=1)

        lstm_layer = rnn.BasicLSTMCell(RNN.n_units, forget_bias=1)
        outputs, _ = rnn.static_rnn(lstm_layer, input, dtype=tf.float32)

        W = tf.Variable(tf.random_normal([RNN.n_units, n_classes]))
        b = tf.Variable(tf.random_normal([n_classes]))

        prediction = tf.matmul(outputs[-1], W) + b
        return prediction
