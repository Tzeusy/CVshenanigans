import numpy as np
import cv2
import os
from pathlib import Path
from coordinate_regressor import get_coordinates, width, height
from mnist_classifier import classify_images, size, num_channels, num_classes

data_source = "data/localization/test"

def localize_and_classify(images):
    coordinates = get_coordinates(images)

    all_crops = []
    for i, image in enumerate(images):
        x, y = coordinates[i]
        offsets = range(-6, 6)

        crops = []
        for i in offsets:
            for j in offsets:
                _x, _y = x+i, y+j
                crop = image[_y-size//2:_y+size//2, _x-size//2:_x+size//2]

                if crop.shape == (size, size):
                    crops.append(np.reshape(crop, (size, size, num_channels)))

        if len(crops) > 0: all_crops.append(crops)

    return [pred for pred, prob in classify_images(all_crops)]

if __name__ == "__main__":
    files = [file for file in os.listdir(data_source) if file.endswith(".png")]
    for label in range(10):
        filenames = [str(Path(data_source, file))for file in files if file[0] == str(label)][:500]
        images = [cv2.imread(name, 0) for name in filenames]

        results = localize_and_classify(images)
        counts = [results.count(i) for i in range(num_classes)]
        print(label, counts, round(max(counts)/sum(counts), 5))
