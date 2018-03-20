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