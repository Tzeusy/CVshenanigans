import numpy as np
import cv2
import os
import beep
from coordinate_regressor import get_coordinates
from mnist_classifier import classify_images

data_source = "./data/localization_data/distributed/training_set/7"
size = 28

def localize_and_classify(image: np.ndarray):
    x, y = get_coordinates(image, display_result=False)

    #offsets = [-28, -14, -7, 0, 7, 14, 28]
    offsets = range(-14, 15, 2)
    crops = []
    for i in offsets:
        for j in offsets:
            _x, _y = (x + i), (y + j)
            crops.append(image[_y:_y+size, _x:_x+size])

    crops = filter(lambda c: c.shape == (size, size), crops)
    crops = [crop.flatten() for crop in crops]
    pred, logit = classify_images(crops)
    return pred

if __name__ == "__main__":
    results = []
    for file in os.listdir(data_source)[:100]:
        if file.endswith(".png"):
            filename = os.path.join(data_source, file)
            print(file, end="\t")
            image = cv2.imread(filename, 0)
            pred = localize_and_classify(image)
            print(pred)
            results.append(pred)
    print(results.count(7), len(results), results.count(0)/len(results))
    beep.beep()
