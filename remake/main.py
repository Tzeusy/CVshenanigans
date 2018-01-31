import numpy as np
import cv2
import os
import beep
from coordinate_regressor import get_coordinates
from mnist_classifier import classify_images

data_source = "./data/localization_data/distributed/test_set"
size = 28

def localize_and_classify(images: list):
    coordinates = get_coordinates(images)

    all_crops = []
    for i, image in enumerate(images):
        y, x = coordinates[i]
        offsets = range(-6, 6)

        crops = []
        for i in offsets:
            for j in offsets:
                _y, _x = (y + i), (x + j)
                crop = image[_y-size//2:_y+size//2, _x-size//2:_x+size//2]
                # ensure that the crop is a 28x28
                if crop.shape == (size, size):
                    crops.append(np.reshape(crop, (size, size, 1)))

        if len(crops) > 0 : all_crops.append(crops)

    return [pred for pred, logit in classify_images(all_crops)]

if __name__ == "__main__":
    for i in map(str, range(10)):
        directory_name = os.path.join(data_source, i)
        files = filter(lambda f: f.endswith(".png"), os.listdir(directory_name))
        images = list(map(lambda f: cv2.imread(os.path.join(directory_name, f), 0), files))

        results = localize_and_classify(images)
        counts = [results.count(i) for i in range(11)]
        print(i, counts, round(max(counts)/sum(counts), 5))
    beep.beep()

