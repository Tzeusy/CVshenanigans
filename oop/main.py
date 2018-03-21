import numpy as np
import tensorflow as tf
import cv2
import os
from pathlib import Path
from exec import Exec

train_data = Path('data/localization/train')
test_data = Path('data/localization/test')

if __name__ == '__main__':
    exec = Exec()

    files = [file for file in os.listdir(train_data) if file.endswith('.png')]
    for label in range(10):
        filenames = [str(train_data / file) for file in files if file[0] == str(label)][:500]
        images = [cv2.imread(name, 0) for name in filenames]

        exec.cnn(label, images)