import tensorflow as tf

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

rx = tf.placeholder(tf.float32, [None,280,280,1])
ry = tf.placeholder(tf.float32,[None,2])

rx_image = tf.reshape(rx, [-1,280,280,1])


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
    xy_distances = tf.abs(y_-y_conv)
    deviation = tf.reduce_mean(xy_distances)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(deviation)

    pixel_distance = tf.sqrt(tf.square(xy_distances[0])+tf.square(xy_distances[1]))
    pixel_distance = tf.cast(pixel_distance,tf.float32)
    accuracy = tf.reduce_mean(pixel_distance)

    return {
        "x": x,
        "y_": y_,
        "keep_prob": keep_prob,
        "y_conv": y_conv,
        "train_step": train_step,
        "accuracy": accuracy
        }
