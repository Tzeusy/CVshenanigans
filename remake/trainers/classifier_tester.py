# https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf
import os
import sys
import time
import numpy as np
from tensorflow.contrib.data import Dataset, Iterator

model_path = "../models/mnist_fc/model.ckpt"
