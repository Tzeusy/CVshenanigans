import os
import random
import time
import cv2
import numpy as np

src = "raw_mnist"
dst = "localization_data"
training = "training_set"
test = "test_set"

DATA_SIZE = 6000
TEST_SIZE = 0.2

BASE_WIDTH = 280
BASE_HEIGHT = 280

def initialize_folders():
    for i in map(str, range(10)):
        os.makedirs(os.path.join(dst, training, i), exist_ok=True)
        os.makedirs(os.path.join(dst, test, i), exist_ok=True)

def make_file(data_type):
    folder_name = os.path.join(src, label)
    image_name = random.choice(os.listdir(folder_name))
    random_image = cv2.imread(os.path.join(folder_name, image_name), 0)

    h,w = random_image.shape[:2]
    x,y = random.randrange(BASE_WIDTH - w), random.randrange(BASE_HEIGHT - h)
    generated_image = np.zeros((BASE_HEIGHT, BASE_WIDTH), np.uint8)
    generated_image[y:y+h, x:x+w] = random_image

    filename = os.path.join(dst, data_type, "{}_{:04d}.png".format(label, i))
    cv2.imwrite(filename, generated_image)

    return y,x

if __name__ == "__main__":
    random.seed(0)
    start_time = time.time()
#    initialize_folders()
    
    training_filename = os.path.join(dst, training, "label.txt")
    test_filename = os.path.join(dst, test, "label.txt")
    with open(training_filename, "w") as training_file, open(test_filename, "w") as test_file:
        for label in map(str, range(10)):
            for i in range(int(DATA_SIZE * (1-TEST_SIZE))):
                y,x = make_file(training)
                training_file.write("{}_{:04d} {} {}\n".format(label, i, y, x))

            for i in range(int(DATA_SIZE * TEST_SIZE)):
                y,x = make_file(test)
                test_file.write("{}_{:04d} {} {}\n".format(label, i, y, x))
    print(time.time() - start_time)
