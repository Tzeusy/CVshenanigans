import tensorflow as tf
from variables import Variable as v
from dataset import Dataset

class Classifier(object):

    def __init__(self, size=28, num_channels=1, num_classes=10):
        self.x = tf.placeholder(tf.float32, shape=[None, size, size, num_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes])

        #self.x_image = tf.reshape(x, [None, size, size, num_channels])
        self.x_image = tf.map_fn(tf.image.per_image_standardization, self.x)

        with tf.variable_scope("classifier") as scope:
            self.output = self.network(self.x_image, size, num_channels, num_classes)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def network(self, x, size, initial_channels, num_classes):
        channels_1 = 32
        channels_2 = 64
        neurons_1 = 1024

        # 2 layers: convolution + max pooling
        h_pool1 = self.conv_layer(x, initial_channels, channels_1)
        h_pool2 = self.conv_layer(h_pool1, channels_1, channels_2)

        # fully connected layer
        new_size = size >> 2 # 2 layers of 2x2 pooling
        fc_length = new_size * new_size * channels_2
        h_fc1 = self.fc_layer(h_pool2, fc_length, neurons_1)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = v.weight_variable([neurons_1, num_classes])
        b_fc2 = v.bias_variable([num_classes])

        # softmax
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_softmax = tf.nn.softmax(y_conv)

        return y_softmax

    def conv_layer(self, inp, input_channels, output_channels):
        W = v.weight_variable([5, 5, input_channels, output_channels])
        b = v.bias_variable([output_channels])

        h = tf.nn.relu(v.conv2d(inp, W) + b)
        h_pool = v.max_pool(h)

        return h_pool

    def fc_layer(self, inp, length, num_neurons):
        W = v.weight_variable([length, num_neurons])
        b = v.bias_variable([num_neurons])

        h_pool_flat = tf.reshape(inp, [-1, length])
        h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W) + b)

        return h_fc

model = '../models/classifier/model.ckpt'
train_data = '../data/mnist/train'
test_data = '../data/mnist/test'

def train():
    num_iterations = 20000

    dataset = Dataset.mnist(train_data)
    iterator = Dataset.iterator(dataset, batch_size=50)

    classifier = Classifier()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(classifier.loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        for i in range(num_iterations):
            x, y = sess.run(next_element)
            feed_dict={
                classifier.x: x,
                classifier.y: y,
                classifier.keep_prob: 0.5
            }

            train_step.run(feed_dict=feed_dict)

            if (i+1) % 100 == 0:
                accuracy = classifier.accuracy.eval(feed_dict=feed_dict)
                print(i+1, accuracy)

        print(saver.save(sess, model))

def test():
    batch_size = 1000
    num_iterations = 10000 // batch_size

    dataset = Dataset.mnist(test_data)
    iterator = Dataset.iterator(dataset, batch_size=batch_size)

    classifier = Classifier()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(classifier.loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model)

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        accuracies = []
        for i in range(num_iterations):
            x, y = sess.run(next_element)
            accuracy = classifier.accuracy.eval(feed_dict={
                classifier.x: x,
                classifier.y: y,
                classifier.keep_prob: 1.0
            })
            accuracies.append(accuracy)

        print(sum(accuracies)/len(accuracies))


if __name__ == '__main__':
    #train()
    test()