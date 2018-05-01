import tensorflow as tf
from functools import partial
from pathlib import Path


# converting data to tensors
def _parse_function(filename, label, width, height):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, (height, width))
    return image_resized, label


def _create_dataset(images, labels, parse):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(-1)
    return dataset


def mnist(src, size=28, num_classes=10):
    src = Path(src)

    image_list = []
    mnist_labels = []
    with open(src / 'label.txt') as f:
        for line in f:
            name, label = line.strip().split(',')
            image_list.append(str(src / name))
            mnist_labels.append(int(label))

    depth = tf.constant(num_classes)
    one_hot_encoded = tf.one_hot(indices=mnist_labels, depth=depth)

    images = tf.constant(image_list)
    labels = one_hot_encoded
    parse = partial(_parse_function, height=size, width=size)

    dataset = _create_dataset(images, labels, parse)
    return dataset


def localization(src, width=280, height=280):
    src = Path(src)

    image_list = []
    coord_labels = []

    with open(src / 'label.txt') as f:
        for line in f:
            name, label, x, y = line.rstrip().split(',')
            image_list.append(str(src / name))
            coord_labels.append((int(x), int(y)))

    images = tf.constant(image_list)
    labels = tf.constant(coord_labels)
    parse = partial(_parse_function, width=width, height=height)

    dataset = _create_dataset(images, labels, parse)
    return dataset


def iterator(dataset, batch_size=50):
    batched_dataset = dataset.batch(batch_size)
    iterator = batched_dataset.make_initializable_iterator()
    return iterator
