import cv2
from pathlib import Path
from tqdm import trange
import params


# Retrieves data from 280x280 data as np.ndarray images
#
# input - dataset: 'train' / 'test'
# output - list of tuples (image: np.ndarray, label: int, coord: tuple)
def load_localization(dataset='train'):
    if dataset == 'train':
        path = Path(params.regressor.train_data)
    elif dataset == 'test':
        path = Path(params.regressor.test_data)
    else:
        raise ValueError('Unknown dataset specified: Use train/test')

    data = []

    # read all information from label.txt file
    with open(path / 'label.txt') as f:
        lines = f.readlines()

    for i in trange(len(lines), desc='Converting files into images'):
        name, label, x, y = lines[i].strip().split(',')

        image = cv2.imread(str(path / name), 0)
        label = int(label)
        coords = (int(x), int(y))

        data.append((image, label, coords))

    return data
