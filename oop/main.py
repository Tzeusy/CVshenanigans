import tensorflow as tf
import os
import params
import data_loader
import model_loader
import inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide warnings


def localize_and_classify(images):
    _offset = params.offset + params.classifier.size // 2

    coords = inference.localize(sess, regressor, images)
    crops = []
    for image, coord in zip(images, coords):
        x, y = coord

        x = max(x, _offset)
        x = min(x, params.regressor.width - _offset)
        y = max(y, _offset)
        y = min(y, params.regressor.height - _offset)

        crop = image[y-_offset: y+_offset, x-_offset: x+_offset]
        crops.append(crop)

    return [pred for pred, prob in inference.rnn_classify(sess, rnn, crops)]


# splits the images by labels and determines the accuracy for each separate label
def cluster_test(batch_size=800):
    data = data_loader.load_localization('test')
    for label in range(10):
        # grab tuples of the corresponding label, then grab the image
        images = [t[0] for t in data if t[1] == label]
        n_batches = len(images)//batch_size + 1

        results = []
        for i in range(n_batches):
            batch = images[i::n_batches]
            results.extend(localize_and_classify(batch))

        counts = [results.count(i) for i in range(params.classifier.n_classes)]
        print(label, counts, round(counts[label] / sum(counts), 5), sep='\t')


if __name__ == '__main__':
    sess = tf.Session()
    regressor = model_loader.load_regressor(sess)
    rnn = model_loader.load_rnn(sess)

    cluster_test()
