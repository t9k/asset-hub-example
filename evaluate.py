import argparse
from datetime import datetime
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir',
                    type=str,
                    default='./data',
                    help='Path of directory of the dataset.')
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
                    default='./metrics',
                    help='Path of output directory of the evaluation metrics.')

logger = logging.getLogger('print')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, params['conv_channels1'],
                               params['conv_kernel_size'], 1)
        self.conv2 = nn.Conv2d(params['conv_channels1'],
                               params['conv_channels2'],
                               params['conv_kernel_size'], 1)
        self.conv3 = nn.Conv2d(params['conv_channels2'],
                               params['conv_channels3'],
                               params['conv_kernel_size'], 1)
        self.pool = nn.MaxPool2d(params['maxpool_size'],
                                 params['maxpool_size'])
        self.dense1 = nn.Linear(576, params['linear_features1'])
        self.dense2 = nn.Linear(params['linear_features1'], 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        output = F.softmax(self.dense2(x), dim=1)
        return output


def test():
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            prediction = output.max(1)[1]
            correct += (prediction == target).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)

    metrics = {'loss': test_loss, 'accuracy': test_accuracy}
    with open(output_path, 'wt') as f:
        f.write(json.dumps(metrics))


if __name__ == '__main__':
    args = parser.parse_args()
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.load_path = os.path.expanduser(args.load_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        datetime.now().strftime('%y%m%d_%H%M%S') + '.json')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        logger.info('NVIDIA_VISIBLE_DEVICES: {}'.format(
            os.getenv('NVIDIA_VISIBLE_DEVICES')))
        logger.info('T9K_GPU_PERCENT: {}'.format(os.getenv('T9K_GPU_PERCENT')))
        logger.info('Device Name {}'.format(torch.cuda.get_device_name()))
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    params = {
        'conv_channels1': 32,
        'conv_channels2': 64,
        'conv_channels3': 64,
        'conv_kernel_size': 3,
        'maxpool_size': 2,
        'linear_features1': 64,
    }

    model = Net().to(device)
    model.load_state_dict(torch.load(args.load_path))
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    test_dataset = datasets.MNIST(root=args.dataset_dir,
                                  train=False,
                                  download=False,
                                  transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1000,
                                              shuffle=False,
                                              **kwargs)

    test()
