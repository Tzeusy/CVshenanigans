import os
import random
import time
import cv2
import numpy as np

src = "raw_mnist"
dst = "localization_data/consolidated"
training = "training_set"
test = "test_set"

DATA_SIZE = 4000
TEST_SIZE = 0.2

BASE_WIDTH = 280
BASE_HEIGHT = 280

def initialize_folders():
    os.makedirs(os.path.join(dst, training), exist_ok=True)
    os.makedirs(os.path.join(dst, test), exist_ok=True)

def make_file(data_type, label):
    folder_name = os.path.join(src, label)
    image_name = random.choice(os.listdir(folder_name))
    random_image = cv2.imread(os.path.join(folder_name, image_name), 0)

    h,w = random_image.shape[:2]
    x,y = random.randrange(BASE_WIDTH - w), random.randrange(BASE_HEIGHT - h)
    generated_image = np.zeros((BASE_HEIGHT, BASE_WIDTH), np.uint8)
    generated_image[y:y+h, x:x+w] = random_image

    filename = os.path.join(dst, data_type, "{}_{:04d}.png".format(label, i))
    cv2.imwrite(filename, generated_image)

    return y,x,h,w

if __name__ == "__main__":
    random.seed(0)
    start_time = time.time()

    initialize_folders()

    training_filename = os.path.join(dst, training, "label.txt")
    test_filename = os.path.join(dst, test, "label.txt")
    with open(training_filename, "w") as f_training, open(test_filename, "w") as f_test:
        for label in map(str, range(10)):
            print(label)
            for i in range(int(DATA_SIZE * (1-TEST_SIZE))):
                y,x,h,w = make_file(training, label)
                f_training.write("{} {:04d} {} {}\n".format(label, i, y+h//2, x+w//2))

            for i in range(int(DATA_SIZE * TEST_SIZE)):
                y,x,h,w = make_file(test, label)
                f_test.write("{} {:04d} {} {}\n".format(label, i, y+h//2, x+w//2))

    print("{:.3f}s".format(time.time() - start_time))
