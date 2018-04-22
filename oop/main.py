import cv2
import os
from pathlib import Path
from tqdm import trange
from exec import Exec

train_data = Path('data/localization/train')
test_data = Path('data/localization/test')


def cluster_test():

    files = [file for file in os.listdir(train_data) if file.endswith('.png')]
    for label in range(10):
        filenames = [str(train_data / file) for file in files if file[0] == str(label)][:800]
        images = [cv2.imread(name, 0) for name in filenames]

        exec.rnn_test(label, images)


def rnn():
    files, labels, coords = [], [], []
    with open(train_data / 'label.txt') as f:
        for line in f:
            name, label, x, y = line.strip().split(',')

            files.append(str(train_data / name))
            labels.append(int(label))
            coords.append((int(x), int(y)))

    images = []
    for i in trange(len(files)):
        images.append(cv2.imread(files[i], 0))

    exec.rnn_train(images, labels, coords)


if __name__ == '__main__':
    exec = Exec()
    # rnn()
    cluster_test()
