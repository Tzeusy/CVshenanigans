import tensorflow as tf
import numpy as np
import os
import time
#import beep

# declaring parameter values
model_path = "../models/localization_regression_consolidated/model.ckpt"
train_data_source = "../data/localization_data/consolidated/training_set"
test_data_source = "../data/localization_data/consolidated/test_set"
image_height, image_width = 280, 280
batch_size = 50
num_iterations = 30000

def get_dataset(data_source):
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_source, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(' ')
            label, index, y, x = map(int, t)
            coord_labels.append((y, x))

    image_list = []
    for file in os.listdir(data_source):
        if file.endswith(".png"):
            image_list.append(os.path.join(data_source, file))

    assert len(coord_labels) == len(image_list), "Mismatch in number of inputs"

    # converting data to tensors
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
        return image_resized, label

    labels = tf.constant(coord_labels)
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
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
train_step = tf.train.AdamOptimizer(1e-4).minimize(deviation)

xy_distances = tf.abs(y_-y_conv)
pixel_distance = tf.sqrt(tf.square(xy_distances[0])+tf.square(xy_distances[1]))
pixel_distance = tf.cast(pixel_distance,tf.float32)
accuracy = tf.reduce_mean(debug)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_dataset = get_dataset(train_data_source)
test_dataset = get_dataset(test_data_source)

with tf.Session() as sess:
    start_time = time.time()
    sess.run(init)

    train_batched_dataset = train_dataset.batch(batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()

    test_batched_dataset = test_dataset.batch(batch_size)
    test_iterator = test_batched_dataset.make_initializable_iterator()
    test_next_element = test_iterator.get_next()

    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)

    for i in range(num_iterations):
        train_iteration = sess.run(train_next_element)
        train_feed_dict = {x: train_iteration[0], y_: train_iteration[1], keep_prob: 0.5}
        sess.run(train_step, feed_dict=train_feed_dict)

        if (i+1) % 100 == 0:
            test_iteration = sess.run(test_next_element)
            test_feed_dict = {x: test_iteration[0], y_:test_iteration[1], keep_prob:1}
            print(i+1, pixel_distance.eval(test_feed_dict))

    file_path = saver.save(sess, model_path)
    print("Model saved in file:", file_path)
    print("Time elapsed: ", time.time()-start_time)
#beep.beep()
# 14
