import tensorflow as tf
from cnn import CNN
from dataset import Dataset

class Regressor(CNN):
    def __init__(self, width=280, height=280, num_channels=1):
        self.x = tf.placeholder(tf.float32, shape=[None, width, height, num_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        #x_image = tf.reshape(self.x, [None, width, height, num_channels])
        self.x_image = tf.map_fn(tf.image.per_image_standardization, self.x)

        with tf.variable_scope("regressor") as scope:
            self.output = self.network(self.x_image, width, height, num_channels)

        diff = tf.abs(self.output - self.y)
        self.loss = tf.reduce_mean(diff)
        self.accuracy = tf.reduce_mean(tf.cast(diff, tf.float32), axis=0)

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

model = '../models/regressor/model.ckpt'
train_data = '../data/localization/train'
test_data = '../data/localization/test'

def train():
    num_iterations = 20000

    dataset = Dataset.localization(train_data)
    iterator = Dataset.iterator(dataset, batch_size=50)

    regressor = Regressor()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(regressor.loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        for i in range(num_iterations):
            x, y = sess.run(next_element)
            feed_dict= {
                regressor.x: x,
                regressor.y: y,
                regressor.keep_prob: 0.5
            }
            train_step.run(feed_dict=feed_dict)

            if (i+1) % 100 == 0:
                accuracy = regressor.accuracy.eval(feed_dict=feed_dict)
                print(i+1, accuracy)

        print(saver.save(sess, model))

if __name__ == '__main__':
    train()