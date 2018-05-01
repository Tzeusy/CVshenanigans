import os
import random
import cv2
import numpy as np
from pathlib import Path

train_src = Path("mnist", "train")
test_src = Path("mnist", "test")
train_dst = Path("localization", "train")
test_dst = Path("localization", "test")

BASE_WIDTH = 280
BASE_HEIGHT = 280

def make_file(dst, name, image):
    h, w = image.shape[:2]
    x, y = random.randrange(BASE_WIDTH - w), random.randrange(BASE_HEIGHT - h)

    new_image = np.zeros((BASE_HEIGHT, BASE_WIDTH), np.uint8)
    new_image[y:y+h, x:x+w] = image

    cv2.imwrite(str(Path(dst, name)), new_image)
    return (x + w//2, y + h//2) # returns the center of the added image

def make_data(src, dst):
    label_file = open(Path(dst, "label.txt"), "w")
    image_files = [file for file in os.listdir(src) if file.endswith(".png")]
    for name in image_files:
        image = cv2.imread(str(Path(src, name)), 0)
        x, y = make_file(dst, name, image)
        label_file.write("{name},{label},{x},{y}\n".format(name=name, label=name[0], x=x, y=y))

    label_file.close()


if __name__ == '__main__':
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    random.seed(0)
    
    make_data(src=train_src, dst=train_dst)
    make_data(src=test_src, dst=test_dst)
