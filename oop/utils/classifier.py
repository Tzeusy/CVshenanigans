import tensorflow as tf
from cnn import CNN
from dataset import Dataset


class Classifier(CNN):
    model = '../models/classifier/model.ckpt'
    train_data = '../data/mnist/train'
    test_data = '../data/mnist/test'

    def __init__(self, size=28, num_channels=1, num_classes=10):
        self.size = size
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.x = tf.placeholder(tf.float32, shape=[None, size, size, num_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes])

        #self.x_image = tf.reshape(x, [None, size, size, num_channels])
        self.x_image = tf.map_fn(tf.image.per_image_standardization, self.x)

        with tf.variable_scope('classifier') as scope:
            self.output = self.network(self.x_image, size, num_channels, num_classes)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        vars = [v for v in tf.global_variables() if v.name.startswith('classifier')]
        self.saver = tf.train.Saver(vars)

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

        W_fc2 = self.weight_variable([neurons_1, num_classes])
        b_fc2 = self.bias_variable([num_classes])

        # softmax
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_softmax = tf.nn.softmax(y_conv)

        return y_softmax


def train():
    classifier = Classifier()
    num_iterations = 20000

    dataset = Dataset.mnist(classifier.train_data)
    iterator = Dataset.iterator(dataset, batch_size=50)

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

            classifier.train_step.run(feed_dict=feed_dict)

            if (i+1) % 100 == 0:
                accuracy = classifier.accuracy.eval(feed_dict=feed_dict)
                print(i+1, accuracy)

        print(saver.save(sess, classifier.model))

def test():
    classifier = Classifier()
    batch_size = 1000
    num_iterations = 10000 // batch_size

    dataset = Dataset.mnist(classifier.test_data)
    iterator = Dataset.iterator(dataset, batch_size=batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, classifier.model)

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