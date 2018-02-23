import tensorflow as tf
import numpy as np
import cv2
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

sys.path.append("../utils/")
from coordinate_regression_variables import get_coordinate_regression_variables
from imageVisualizer import show_image_with_crosshairs
from classifier_methods import classifyNumpyArr

# declaring parameter values
regression_model_path = "../models/localization_regression_consolidated/model.ckpt"
regress_data_source = "../data/localization_data/consolidated/test_set"

# converting data to tensors
def _parse_function(filename, label):
    image_height, image_width = 280, 280
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=1)
    image_resized = tf.image.resize_images(image_decoded, [image_height, image_width])
    return image_resized, label, filename

def get_dataset(label):
    if(label==""):
        data_path = regress_data_source
    else: data_path = os.path.join(regress_data_source, str(label))

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

def localize_images():
    r = get_coordinate_regression_variables()
    regress_x, regress_y_, regress_keep_prob = r["x"], r["y_"], r["keep_prob"]
    regress_y_conv, regress_train_step, regress_accuracy = r["y_conv"], r["train_step"], r["accuracy"]

    init = tf.global_variables_initializer()

    saver2 = tf.train.Saver()

    segmented_images = []

    with tf.Session() as sess:
        sess.run(init)
        saver2.restore(sess, regression_model_path)
        dataset = get_dataset("")
        #we use 1 so humans can see
        batched_dataset = dataset.batch(1)
        iterator = batched_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)
        for i in range(100):
            iteration = sess.run(next_element)
            feed_dict = {regress_x: iteration[0], regress_y_: iteration[1], regress_keep_prob: 1.0}
            predicted_value = regress_y_conv.eval(feed_dict)
            print("{},{}".format(i,predicted_value))
            # show_image_with_crosshairs(iteration,predicted_value)
            segmented_images.append((iteration,predicted_value))
        sess.close()
        return segmented_images

if __name__ == "__main__":
    segmented_images = localize_images()
    seed=98
    count=0
    print(len(segmented_images))
    for segmentImage in segmented_images:
        count+=1
        if(count<seed):
            continue
        # tf.reset_default_graph()
        tensorObject = segmentImage[0]
        predicted_coordinates = segmentImage[1][0]

        crops = []
        fullImage = cv2.imread(segmentImage[0][2][0].decode('ascii'),0)

        for i in range(8):
            for j in range(8):
                starty = int(round(predicted_coordinates[0]-18+i*2))
                endy = int(round(predicted_coordinates[0]+10+i*2))
                startx = int(round(predicted_coordinates[1]-18+j*2))
                endx = int(round(predicted_coordinates[1]+10+j*2))
                crop = fullImage[starty:endy,startx:endx]
                if(crop.shape!=(28,28)):
                    continue
                crops.append(crop)

        # show_image_with_crosshairs(segmentImage[0],segmentImage[1])
        predictions = np.zeros(11)
        talliedScores = np.zeros(11)
        highestScores = np.zeros(11)
        for crop in crops:
            # tf.reset_default_graph()
            information = classifyNumpyArr(crop)
            predictions[information[0]]+=1
            for i in range(11):
                talliedScores[i]+=information[1][i]
                if(highestScores[i]<information[1][i]):
                    highestScores[i]=information[1][i]

        print("After Image, net Predicitons are {}".format(predictions))
        print("After Image, net Weights are {}".format(talliedScores))
        print("After Image, max Weights are {}".format(talliedScores))
