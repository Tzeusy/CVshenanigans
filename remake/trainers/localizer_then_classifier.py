import tensorflow as tf
import os
import sys
import time
import numpy as np
from tensorflow.contrib.data import Dataset, Iterator
from imageVisualizer import show_small_image

sys.path.append("../utils/")
from coordinate_regression_variables import get_coordinate_regression_variables
