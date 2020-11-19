from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# from models.wideresnet import *
from models.wideresnet_BiaMAT import *
from dataset import *
from trades import *
import time
import pickle

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=110, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-schedule', type=str, default='bag_of_tricks',
                    choices=('trades', 'bag_of_tricks'),
                    help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='a hyperparameter for BiaMAT')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', type=str,
                    help='directory of model for saving checkpoint')
parser.add_argument('--data-dir', type=str, default='data',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--aux-dataset-dir', default='Imagenet32_train')
parser.add_argument('--load-model-dir',
                   help='directory of model for saving checkpoint')
parser.add_argument('--load-epoch', type=int, default=0, metavar='N',
                    help='load epoch')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print(args)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def _load_datafile(data_file, img_size=32):
    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    y = np.array(y)
    assert x.dtype == np.uint8
    return x, y

train_filenames = ['train_data_batch_{}'.format(ii + 1) for ii in range(10)]
x_list = []
y_list = []
for ii, fname in enumerate(train_filenames):
    cur_images, cur_labels = _load_datafile(os.path.join(args.aux_dataset_dir, fname))
    x_list.append(cur_images)
    y_list.append(cur_labels)
data_imagenet = np.concatenate(x_list, axis=0)
label_imagenet = np.concatenate(y_list, axis=0)

print('dataset size : ', data_imagenet.shape, label_imagenet.shape)
print('dataset min, max : ', np.min(data_imagenet), np.max(data_imagenet)) # 0 255
print('label min, max : ', np.min(label_imagenet), np.max(label_imagenet)) # 0 999
print("aug param : ", args.alpha)

n_class_aux = 1000
cur_trainset = ImageNet32(data_imagenet, label_imagenet, transform_train) # 0, 125
train_loader_aug = torch.utils.data.DataLoader(cur_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta
                          )
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train_BiaMAT(args, model, device, train_loader, train_loader_aug, aug_iterator, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            data_aug, target_aug = next(aug_iterator)
        except StopIteration:
            aug_iterator = iter(train_loader_aug)
            data_aug, target_aug = next(aug_iterator)
        split = [len(data), len(data_aug)]
        data, target = torch.cat((data, data_aug), dim=0), torch.cat((target, target_aug), dim=0)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # calculate robust loss
        loss, loss_aux = trades_loss_BiaMAT(model=model,
                           x_natural=data,
                           y=target,
                           split=split,
                           n_class_aux=n_class_aux,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           alpha=args.alpha
                          )
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss Multihead: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_aux.item()))
    return aug_iterator

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if args.lr_schedule == 'trades':
        if epoch >= int(args.epochs * 0.75):
            lr = args.lr * 0.1
        if epoch >= int(args.epochs) * 0.9:
            lr = args.lr * 0.01
    elif args.lr_schedule == 'bag_of_tricks':
        if epoch >= int(args.epochs * (100./110.)):
            lr = args.lr * 0.1
        if epoch >= int(args.epochs * (105./110.)):
            lr = args.lr * 0.01
    else:
        print('specify lr scheduler!!')
        sys.exit(1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model
    model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.load_epoch != 0:
        print('resuming ... ', args.load_model_dir)
        f_path = os.path.join(args.load_model_dir)
        checkpoint = torch.load(f_path)
        model.load_state_dict(checkpoint)
        eval_test(model, device, test_loader)

    init_time = time.time()

    aug_iterator = iter(train_loader_aug)
    for epoch in range(args.load_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        elapsed_time = time.time() - init_time
        print('elapsed time : %d h %d m %d s' % (elapsed_time / 3600, (elapsed_time % 3600) / 60, (elapsed_time % 60)))
        aug_iterator = train_BiaMAT(args, model, device, train_loader, train_loader_aug, aug_iterator, optimizer, epoch)

        # save checkpoint
        if epoch <= 10:
            eval_test(model, device, test_loader)

        if (epoch % args.save_freq == 0) and (epoch >= int(args.epochs * 0.9)):
            # evaluation on natural examples
            print('================================================================')
            eval_test(model, device, test_loader)
            print('================================================================')
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
        torch.save(model.state_dict(),
                   os.path.join(model_dir, 'model-wideres-latest.pt'.format(epoch)))



if __name__ == '__main__':
    main()
