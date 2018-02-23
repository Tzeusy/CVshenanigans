import numpy as np
import cv2
import os
import beep
from coordinate_regressor import get_coordinates
from mnist_classifier import classify_images

data_source = "./data/localization_data/distributed/test_set"
size = 28

def localize_and_classify(images: list, label: int):
    coordinates = get_coordinates(images)

    all_crops = []
    for idx, image in enumerate(images):
        y, x = coordinates[idx]
        offsets = range(-6, 6)

        crops = []
        for i in offsets:
            for j in offsets:
                _y, _x = (y + i), (x + j)
                crop = image[_y-size//2:_y+size//2, _x-size//2:_x+size//2]
                
                # ensure that the crop is a 28x28
                if crop.shape == (size, size):
                    crops.append(np.reshape(crop, (size*size,)))

        if len(crops) > 0 : all_crops.append(crops)
    return [pred for pred, logit in classify_images(all_crops, label)]

if __name__ == "__main__":
    for i in map(str, range(10)):
        directory_name = os.path.join(data_source, i)
        files = [f for f in os.listdir(directory_name) if f.endswith(".png")]
        images = [cv2.imread(os.path.join(directory_name, f), 0) for f in files]

        results = localize_and_classify(images, label=int(i))
        counts = [results.count(i) for i in range(10)]
        print(i, counts, round(max(counts)/sum(counts), 5))

        #files = filter(lambda f: f.endswith(".png"), os.listdir(directory_name)[800:])
        #images = list(map(lambda f: cv2.imread(os.path.join(directory_name, f), 0), files))

        #results = localize_and_classify(images, label=int(i))
        #counts = [results.count(i) for i in range(10)]
        #print(i, counts, round(max(counts)/sum(counts), 5))
    beep.beep()
""" mnist 6x6 crops
0 [339, 2, 22, 0, 16, 2, 5, 10, 2, 0] 0.85176
1 [6, 285, 6, 1, 35, 0, 48, 15, 2, 2] 0.7125
2 [3, 2, 379, 3, 0, 1, 3, 9, 0, 0] 0.9475
3 [0, 0, 36, 348, 0, 5, 0, 9, 1, 0] 0.87218
4 [5, 1, 18, 2, 355, 0, 10, 4, 3, 2] 0.8875
5 [3, 2, 52, 11, 3, 314, 3, 8, 0, 0] 0.79293
6 [0, 9, 19, 0, 20, 0, 346, 2, 1, 2] 0.86717
7 [0, 5, 36, 2, 2, 12, 4, 336, 0, 1] 0.84422
8 [2, 0, 34, 2, 13, 3, 16, 8, 319, 2] 0.7995
9 [1, 1, 20, 0, 6, 0, 46, 1, 4, 320] 0.80201
"""
""" crops only 1x1 crop
0 [357, 5, 6, 1, 6, 3, 9, 4, 1, 0] 0.91071
1 [10, 362, 5, 0, 9, 1, 5, 4, 0, 0] 0.91414
2 [22, 2, 325, 1, 9, 4, 3, 18, 5, 1] 0.83333
3 [20, 2, 35, 284, 4, 11, 2, 19, 11, 5] 0.72265
4 [8, 2, 0, 0, 353, 3, 9, 15, 0, 5] 0.89367
5 [27, 3, 8, 4, 8, 314, 18, 5, 3, 4] 0.79695
6 [17, 10, 0, 0, 6, 0, 353, 0, 0, 7] 0.89822
7 [10, 6, 8, 0, 8, 4, 3, 351, 0, 6] 0.88636
8 [41, 3, 6, 2, 7, 5, 21, 7, 286, 20] 0.71859
9 [8, 3, 2, 1, 30, 3, 5, 15, 4, 325] 0.82071
"""
""" crops only 6x6 crop
0 [384, 0, 2, 0, 5, 0, 5, 2, 0, 0] 0.96482
1 [9, 382, 5, 0, 1, 1, 0, 2, 0, 0] 0.955
2 [17, 2, 344, 0, 7, 7, 4, 17, 1, 1] 0.86
3 [21, 1, 32, 299, 6, 8, 3, 19, 3, 7] 0.74937
4 [18, 1, 1, 0, 369, 0, 5, 4, 0, 2] 0.9225
5 [19, 1, 4, 3, 4, 344, 14, 5, 1, 1] 0.86869
6 [8, 6, 0, 0, 2, 0, 381, 0, 0, 2] 0.95489
7 [2, 1, 4, 0, 4, 2, 1, 382, 0, 2] 0.9598
8 [36, 3, 6, 0, 5, 5, 10, 5, 322, 7] 0.80702
9 [17, 0, 3, 0, 18, 0, 9, 9, 0, 343] 0.85965
"""
""" crops & mnist 6x6 crop
0 [367, 8, 6, 0, 4, 3, 0, 8, 0, 2] 0.92211
1 [0, 381, 4, 0, 0, 1, 2, 9, 1, 2] 0.9525
2 [3, 1, 371, 4, 2, 8, 1, 9, 1, 0] 0.9275
3 [1, 0, 9, 374, 1, 9, 1, 4, 0, 0] 0.93734
4 [5, 1, 7, 0, 378, 0, 3, 4, 0, 2] 0.945
5 [3, 1, 16, 4, 1, 362, 2, 6, 0, 1] 0.91414
6 [2, 13, 1, 0, 5, 1, 373, 2, 0, 2] 0.93484
7 [0, 4, 7, 2, 0, 9, 0, 376, 0, 0] 0.94472
8 [4, 3, 9, 4, 1, 10, 9, 7, 348, 4] 0.87218
9 [1, 0, 9, 1, 12, 2, 1, 4, 0, 369] 0.92481
"""
