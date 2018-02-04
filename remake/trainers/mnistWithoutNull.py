# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
import os
import sys
import time

sys.path.append("../utils/")
from mnist_classification_without_null_variables import get_mnist_classification_variables

model_path = "../models/mnist_fc_without_null/model.ckpt"
train_data_source = "../data/raw_mnist/jointTraining"
test_data_source = "../data/raw_mnist/jointTest"
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

v = get_mnist_classification_variables()
x, y, keep_prob = v["x"], v["y"], v["keep_prob"]
y_conv, train_step, accuracy = v["y_conv"], v["train_step"], v["accuracy"]

init = tf.global_variables_initializer()
saver = tf.train.Saver()

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
        
        #test_iteration = sess.run(test_next_element)
        #test_feed_dict = {x: test_iteration[0], y:test_iteration[1], keep_prob:1}
        #acc+=accuracy.eval(test_feed_dict)

        if (i+1) % 100 == 0:
            print("Cycle {}".format(i+1))
            print("Last 100 batches have an accuracy of: ")
            print(acc/100)
            acc = 0

    file_path = saver.save(sess, model_path)
    print("Model saved in file:", file_path)
    print("Time elapsed: ", time.time()-start_time)
