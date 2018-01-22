import numpy as np
import cv2
import os
import beep
from coordinate_regressor import get_coordinates
from mnist_classifier import classify_images

data_source = "./data/localization_data/distributed/training_set/1"
size = 28

def localize_and_classify(image: np.ndarray):
    x, y = get_coordinates(image, display_result=False)

    offsets = range(-14, 15)
    crops = []
    for i in offsets:
        for j in offsets:
            _x, _y = (x + i), (y + j)
            crop = image[_y-size//2:_y+size//2, _x-size//2:_x+size//2]
            crops.append(crop)

    crops = filter(lambda c: c.shape == (size, size), crops)
    crops = [crop.flatten() for crop in crops]
    pred, logit = classify_images(crops)
    return pred

if __name__ == "__main__":
    results = []
    for file in os.listdir(data_source)[:500]:
        if file.endswith(".png"):
            filename = os.path.join(data_source, file)
            print(file, end="\t")
            image = cv2.imread(filename, 0)
            pred = localize_and_classify(image)
            print(pred)
            results.append(pred)
    print(results.count(1), len(results), results.count(1)/len(results))
    print([results.count(i) for i in range(10)])
    beep.beep()
# 2x2 - [396, 0, 30, 0, 13, 22, 1, 36, 0, 2]
# 3x3 - [378, 2, 14, 0, 44, 57, 1, 1, 2, 1]
# 4x4 - [418, 2, 5, 1, 16, 16, 0, 42, 0, 0]
# 5x5 - [294, 41, 26, 0, 6, 109, 0, 22, 0, 2]
