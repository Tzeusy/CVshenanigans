# https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import os
import time

model_path = "../models/mnist_fc/model.ckpt"
train_data_source = "../data/raw_mnist/jointNoNullTrain"
test_data_source = "../data/raw_mnist/jointNoNullTest"
image_height, image_width = 28, 28
batch_size = 50
num_iterations = 30000

def labelToInt(x):
    try:
        return int(x)
    except:
        return(10)

def get_dataset(data_source):
    # parsing data from file
    coord_labels = []
    with open(os.path.join(data_source, "label.txt")) as f:
        for line in f:
            t = line.rstrip().split(', ')
            filename, value = map(labelToInt, t)
            coord_labels.append(value)
            
    depth = tf.constant(10)
    one_hot_encoded = tf.one_hot(indices=coord_labels, depth=depth)

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

    labels = one_hot_encoded
    images = tf.constant(image_list)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(-1)

    return dataset

# here we begin creation of a new model, involving cnn
# weights are initialized with a small amount of noise to break symmetry, and prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# since neurons are ReLU, initialize them with a slight positive bias to avoid "dead neurons"
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# x is the images, while y is the result
x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
y = tf.placeholder(tf.float32, shape=[None, 10])

# implementation of the first layer
# compute 32 features for each 5x5 patch
W_conv1 = weight_variable([5,5,1,32]) # (patch_size, patch_size, # input channels, # output channels)
b_conv1 = bias_variable([32])
# reshape x to a 4d tensor, with 2nd and 3rd dimensions being width and height, and 4th dimension being number of colour channels
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x_image with W, then add b, apply the ReLU function, then max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # this reduces image size to 14x14

# implementation of the second layer
# 64 features for each 5x5
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # 7x7

# implementation of fully-connected layer with 1024 neurons - this processes based on the entire image
W_fc1 = weight_variable([7*7 * 64, 1024]) # 64 layers of 7x7 image
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) # flatten the image into a single line
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout - reduces overfitting
# dropout randomly selects units in a neural network, and temporarily removes all incoming and outgoing connections
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# layer for softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# differences from beginner
#   1. replace gradient descent with ADAM
#   2. include keep_prob in the parameters of feed_dict to control dropout rate
#   3. add logging to every 100th iteration in the process
#   4. use tf.Session instead of tf.InteractiveSession - separates the process of creating the graph from evaluating the graph

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
#saver = tf.train.Saver()

train_dataset = get_dataset(train_data_source)
test_dataset = get_dataset(test_data_source)

with tf.Session() as sess:
    sess.run(init)

    iterations = 0
    start_time = time.time()
    
    train_batched_dataset = train_dataset.batch(batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()

    test_batched_dataset = test_dataset.batch(batch_size)
    test_iterator = test_batched_dataset.make_initializable_iterator()
    test_next_element = test_iterator.get_next()

    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    
    acc = 0
    
    for i in range(num_iterations):
        
        train_iteration = sess.run(train_next_element)
        train_feed_dict = {x: train_iteration[0], y: train_iteration[1], keep_prob: 0.5}
        sess.run(train_step, feed_dict=train_feed_dict)
        
        test_iteration = sess.run(test_next_element)
        test_feed_dict = {x: test_iteration[0], y:test_iteration[1], keep_prob:1}
        acc+=accuracy.eval(test_feed_dict)

        if (i+1) % 100 == 0:
            print("Cycle {}".format(i))
            print("Last 100 batches have an accuracy of: ")
            print(acc/100)
            acc = 0

#    file_path = saver.save(sess, model_path)
#    print("Model saved in file:", file_path)
    print("Time elapsed: ", time.time()-start_time)
 
