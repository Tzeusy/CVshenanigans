import os
from pathlib import Path

# ensure all paths in params are relative to this file
root = Path(os.path.abspath(os.path.dirname(__file__)))

offset = 12


class regressor:
    model_path = str(root / 'models/regressor/model.ckpt')
    train_data = str(root / 'data/localization/train')
    test_data = str(root / 'data/localization/test')

    width = 280
    height = 280
    n_channels = 1


class classifier:
    model_path = str(root / 'models/classifier/model.ckpt')
    train_data = str(root / 'data/mnist/train')
    test_data = str(root / 'data/mnist/test')

    size = 28
    n_channels = 1
    n_classes = 10


class rnn:
    model_path = str(root / 'models/rnn_classifier/model.ckpt')
    n_classes = 10
