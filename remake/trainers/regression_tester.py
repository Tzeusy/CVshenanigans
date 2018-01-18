import tensorflow as tf
import numpy as np
import cv2
import os

# declaring parameter values
model_path = "../models/localization_regression/model.ckpt"
data_source = "../data/localization_data/distributed/test_set"
image_height, image_width = 280, 280
batch_size = 50

# converting data to tensors
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
    return image_resized, label

def get_dataset(label):
    data_path = os.path.join(data_source, str(label))
    
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_path, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(' ')
            label, index, y, x = map(int, t)
            coord_labels.append((y, x))

    image_list = []
    for file in os.listdir(data_path):
        if file.endswith(".png"):
            image_list.append(os.path.join(data_path, file))

    assert len(coord_labels) == len(image_list), "Mismatch in number of inputs"

    labels = tf.constant(coord_labels)
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(-1)

    return dataset

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

x_image=tf.reshape(x,[-1,280,280,1])
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

debug = tf.abs(y_-y_conv)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

accumulated_accuracy = 0.
total_iterations = 0
    
for i in range(10):
    print("Label:", i)
    dataset = get_dataset(i)
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)

        for i in range(10):
            iteration = sess.run(next_element)
            feed_dict = {x: iteration[0], y_: iteration[1], keep_prob: 1.0}
            sess.run(train_step, feed_dict=feed_dict)

            test_accuracy = accuracy.eval(feed_dict)
            print(i, accuracy.eval(feed_dict))
            accumulated_accuracy+= test_accuracy
            total_iterations+= 1

print("Total average accuracy:", accumulated_accuracy/total_iterations)
