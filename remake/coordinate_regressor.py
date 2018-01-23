import tensorflow as tf
import numpy as np
import cv2
import os

def get_coordinates(images: np.ndarray):
    # declaring parameter values
    model_path = "./models/localization_regression_consolidated/model.ckpt"
    image_height, image_width = 280, 280
    size = 28
    tf.reset_default_graph()

    # defining variables
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    x = tf.placeholder(tf.float32, [None,280,280,1])
    y_= tf.placeholder(tf.float32,[None,2])

    x_image=tf.reshape(x,[-1,image_width,image_height,1])
    ##############
    h_pool0 = max_pool_2x2(x_image)
    h_pool5 = max_pool_2x2(h_pool0)

    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(h_pool5,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([18*18*64,1024])
    b_fc1= bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool2,[-1,18*18*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    W_fc2 = weight_variable([1024,2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

    # loss function
    deviation = tf.reduce_mean(tf.abs(y_-y_conv))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(deviation)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    tf_images = [image.reshape((image_height, image_width, 1)) for image in images]
    labels = [np.array([0,0]) for i in range(len(images))]

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        feed_dict = {x: tf_images, y_: labels, keep_prob: 1.0}
        result = sess.run(y_conv, feed_dict)
        return [tuple(map(int, t)) for t in result]

if __name__ == "__main__":
    data_source = "./data/localization_data/distributed/test_set"
    data_path = os.path.join(data_source, "0")
    image_filename = os.path.join(data_path, "0_0000.png")
    image = cv2.imread(image_filename, 0)
    print(get_coordinates([image, image]))
