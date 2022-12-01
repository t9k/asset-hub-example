import argparse
import json
import logging
import gzip
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import callbacks, datasets, layers, models, optimizers

def convert_to_numpy(data_dir, images_file, labels_file):
    """Byte string to numpy arrays"""
    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = np.frombuffer(f.read(), np.uint8,
                               offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return images, labels


def mnist_to_numpy(data_dir, train):
    """Load raw MNIST data into numpy array

    Args:
        data_dir (str): directory of MNIST raw data.
            This argument can be accessed via SM_CHANNEL_TRAINING

        train (bool): use training data

    Returns:
        tuple of images and labels as numpy array
    """

    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    return convert_to_numpy(data_dir, images_file, labels_file)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir',
                    type=str,
                    default='./data',
                    help='Path of the dataset.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
parser.add_argument('--load_path',
                    type=str,
                    default=None,
                    help='Load path of the trained model.')
parser.add_argument('--output_dir',
                    type=str,
                    default='./results',
                    help='Path of output directory of the evaluation metrics.')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.load_path = os.path.expanduser(args.load_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'metrics.json')

    if args.no_cuda:
        tf.config.set_visible_devices([], 'GPU')

    logger = logging.getLogger('print')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False

    model = models.load_model(args.load_path)

    dataset_dir=os.path.join(args.dataset_dir,'MNIST','raw')
    test_images, test_labels = mnist_to_numpy(data_dir=dataset_dir, train=False)

    test_images = test_images.reshape((10000, 28, 28, 1))

    test_images = test_images / 255.0

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

    metrics = {'loss': test_loss, 'accuracy': test_accuracy}

    output_path = os.path.join(args.output_dir, 'metrics.json')
    with open(output_path, 'wt') as f:
        f.write(json.dumps(metrics))