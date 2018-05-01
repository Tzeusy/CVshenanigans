import sys
sys.path.insert(0, './utils')
import params
from regressor import Regressor
from classifier import Classifier
from rnn import RNN


def load_regressor(sess):
    p = params.regressor
    regressor = Regressor(p.height, p.width, p.n_channels)
    regressor.saver.restore(sess, p.model_path)
    return regressor


def load_classifier(sess):
    p = params.classifier
    classifier = Classifier(p.size, p.n_channels, p.n_classes)
    classifier.saver.restore(sess, p.model_path)
    return classifier


def load_rnn(sess):
    size = params.classifier.size + params.offset * 2
    rnn = RNN(time_steps=size, n_input=size, n_classes=params.rnn.n_classes)
    rnn.saver.restore(sess, params.rnn.model_path)
    return rnn
