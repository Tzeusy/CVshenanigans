import cv2
import os
from pathlib import Path
from exec import Exec

train_data = Path('data/localization/train')
test_data = Path('data/localization/test')


def normal_cnn():
    exec = Exec()

    files = [file for file in os.listdir(train_data) if file.endswith('.png')]
    for label in range(10):
        filenames = [str(train_data / file) for file in files if file[0] == str(label)][:800]
        images = [cv2.imread(name, 0) for name in filenames]

        exec.cnn(label, images)


def rnn():
    files, labels, coords = [], [], []
    with open(train_data / 'label.txt') as f:
        for line in f:
            name, label, x, y = line.strip().split(',')

            files.append(str(train_data / name))
            labels.append(int(label))
            coords.append((int(x), int(y)))

    images = []
    for i, file in enumerate(files):
        images.append(cv2.imread(file, 0))
        if i % 1000 == 0:
            print(i)

    exec = Exec()
    exec.rnn_train(images, labels, coords)


if __name__ == '__main__':
    #normal_cnn()
    rnn()