from torchvision import datasets, transforms
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import sys
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import utils
from torchvision.datasets import CIFAR10, CIFAR100

def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1, 1, 1)),
    ])
    cifar_train = CIFAR10("./data", train=True, download=True, transform=transform_train)
    cifar_test = CIFAR10("./data", train=False, download=True, transform=transform_test)
    return cifar_train, cifar_test

def get_tiny_imagenet_train_valid_loader():

    root = './data'
    tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
    tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
    transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
    trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200-fixed/train',
                                                transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200-fixed/val',
                                               transform=transform_test)

    return trainset, testset

def get_cifar100_dataset():
    """returns trainset and testsets for Fashion CIFAR10 dataset"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    return trainset, testset

def get_dataset(args):
    if args.dataset_name =='cifar10':
        train_set, test_set = get_cifar10()
    elif args.dataset_name == 'cifar100':
        train_set, test_set = get_cifar100_dataset()
    elif args.dataset_name == 'imagenet':
        train_set, test_set = get_tiny_imagenet_train_valid_loader()
    else:
        raise NotImplementedError('Wrong dataset name son.')
    return train_set,test_set
def get_data_loader(args,test_bs=100):
    train_set, test_set = get_dataset(args)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False, pin_memory=True)
    return trainloader,testloader