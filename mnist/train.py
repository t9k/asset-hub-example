import argparse
import logging
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(
    description='Distributed training of PyTorch model for MNIST with DDP.')
parser.add_argument('--aimd',
                    action='store_true',
                    help='Use AIMD to record training data.')
parser.add_argument('--api_key',
                    type=str,
                    help='API Key for requesting AIMD server. '
                    'Required if --aimd is set.')
parser.add_argument(
    '--backend',
    type=str,
    help='Distributed backend',
    choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
    default=dist.Backend.GLOO)
parser.add_argument('--dataset_dir',
                    type=str,
                    default='./data',
                    help='Path of the dataset.')
parser.add_argument(
    '--folder_path',
    type=str,
    default='aimd-example',
    help='Path of AIMD folder in which trial is to be created.')
parser.add_argument(
    '--host',
    type=str,
    default='https://home.nsfocus.t9kcloud.cn/t9k/aimd/server',
    help='URL of AIMD server. Required if --aimd is set.')
parser.add_argument('--log_dir',
                    type=str,
                    default=None,
                    help='Path of the TensorBoard log directory.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
parser.add_argument('--save_path',
                    type=str,
                    default=None,
                    help='Save path of the trained model.')
parser.add_argument('--trial_name',
                    type=str,
                    default='mnist_torch_distributed',
                    help='Name of AIMD trial to create.')

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


def train(scheduler):
    global global_step
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if step % (500 // world_size) == 0:
                train_loss = loss.item()
                logger.info(
                    'epoch {:d}/{:d}, batch {:5d}/{:d} with loss: {:.4f}'.
                    format(epoch, epochs, step, steps_per_epoch, train_loss))
                global_step = (epoch - 1) * steps_per_epoch + step

                if args.log_dir and rank == 0:
                    writer.add_scalar('train/loss', train_loss, global_step)

                if args.aimd and rank == 0:
                    trial.log(
                        metrics_type='train',  # 记录训练指标
                        metrics={'loss': train_loss},  # 指标名称及相应值
                        step=global_step,  # 当前全局步数
                        epoch=epoch)  # 当前回合数

        scheduler.step()
        global_step = epoch * steps_per_epoch
        test(val=True, epoch=epoch)


def test(val=False, epoch=None):
    label = 'val' if val else 'test'
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        loader = val_loader if val else test_loader
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            prediction = output.max(1)[1]
            correct += (prediction == target).sum().item()

    test_loss = running_loss / len(loader)
    test_accuracy = correct / len(loader.dataset)
    msg = '{:s} loss: {:.4f}, {:s} accuracy: {:.4f}'.format(
        label, test_loss, label, test_accuracy)
    if val:
        msg = 'epoch {:d}/{:d} with '.format(epoch, epochs) + msg
    logger.info(msg)

    if args.log_dir and rank == 0:
        writer.add_scalar('{:s}/loss'.format(label), test_loss, global_step)
        writer.add_scalar('{:s}/accuracy'.format(label), test_accuracy,
                          global_step)

    if args.aimd and rank == 0:
        trial.log(
            metrics_type=label,  # 记录验证/测试指标
            metrics={
                'loss': test_loss,
                'accuracy': test_accuracy,
            },
            step=global_step,
            epoch=epoch)


if __name__ == '__main__':
    args = parser.parse_args()
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.save_path = os.path.expanduser(args.save_path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        logger.info('NVIDIA_VISIBLE_DEVICES: {}'.format(
            os.getenv('NVIDIA_VISIBLE_DEVICES')))
        logger.info('T9K_GPU_PERCENT: {}'.format(os.getenv('T9K_GPU_PERCENT')))
        logger.info('Device Name {}'.format(torch.cuda.get_device_name()))
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    logger.info('Using distributed PyTorch with {} backend'.format(
        args.backend))
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    params = {
        'batch_size': 32 * world_size,
        'epochs': 10,
        'learning_rate': 0.001 * world_size,
        'conv_channels1': 32,
        'conv_channels2': 64,
        'conv_channels3': 64,
        'conv_kernel_size': 3,
        'maxpool_size': 2,
        'linear_features1': 64,
        'seed': 1,
    }

    torch.manual_seed(params['seed'])

    model = Net().to(device)
    model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    if args.aimd and rank == 0:
        from t9k import aimd
        aimd.login(host=args.host, api_key=args.api_key)

        trial = aimd.create_trial(trial_name=args.trial_name,
                                  folder_path=args.folder_path)
        trial.params.update(params)
        trial.params.parse(dist_torch_model=model)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    train_dataset = datasets.MNIST(root=args.dataset_dir,
                                   train=True,
                                   download=False,
                                   transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [48000, 12000])
    test_dataset = datasets.MNIST(root=args.dataset_dir,
                                  train=False,
                                  download=False,
                                  transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params['batch_size'],
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=400,
                                             shuffle=False,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1000,
                                              shuffle=False,
                                              **kwargs)

    if args.log_dir and rank == 0:
        log_dir = args.log_dir
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)
        writer = SummaryWriter(log_dir)

    global_step = 0
    epochs = params['epochs']
    steps_per_epoch = len(train_loader)
    train(scheduler)
    test()

    if args.save_path and rank == 0:
        dirname = os.path.dirname(args.save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(model.module.state_dict(), args.save_path)

    if args.aimd and rank == 0:
        trial.finish()
        trial.upload()
