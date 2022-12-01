import argparse
import json
import gzip
import numpy as np
import os
import shutil

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


parser = argparse.ArgumentParser(
    description='Distributed training of Keras model with MultiWorkerMirroredStrategy.')
parser.add_argument('--log_dir',
                    type=str,
                    help='Path of the tensorboard log directory.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
parser.add_argument('--save_path',
                    type=str,
                    help='Save path of the trained model.')
parser.add_argument('--dataset_dir',
                    type=str,
                    default='./dataset',
                    help='Path of the dataset.')
parser.add_argument('--load_path',
                    type=str,
                    default=None,
                    help='Load path of the trained model used to continue training.')
args = parser.parse_args()

if args.no_cuda:
    tf.config.set_visible_devices([], 'GPU')

strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Get world size and index of current worker.
tf_config = json.loads(os.environ['TF_CONFIG'])
world_size = len(tf_config['cluster']['worker'])
task_index = tf_config['task']['index']

with strategy.scope():
    if args.load_path:
        model = models.load_model(args.load_path)
    else:
        model = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ])
        
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001 * world_size),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

dataset_dir=os.path.join(args.dataset_dir,'MNIST','raw')
train_images, train_labels = mnist_to_numpy(data_dir=dataset_dir, train=True)

train_images = train_images.reshape((60000, 28, 28, 1))

train_images = train_images / 255.0

# Set save path for TensorBoard log.
train_callbacks = []
if task_index == 0:
    log_dir = args.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True)
    train_callbacks.append(callbacks.TensorBoard(log_dir=log_dir))

model.fit(train_images,
          train_labels,
          batch_size=32 * world_size,
          epochs=10,
          validation_split=0.2,
          callbacks=train_callbacks,
          verbose=2)

# Set save path and save model.
if task_index == 0:
    save_path = args.save_path
else:
    dirname = os.path.dirname(args.save_path)
    basename = os.path.basename(
        args.save_path) + '_temp_' + str(task_index)
    save_path = os.path.join(dirname, basename)
if os.path.exists(save_path):
    shutil.rmtree(save_path, ignore_errors=True)
# save model
model.save(save_path)
if task_index != 0:
    shutil.rmtree(save_path, ignore_errors=True)
