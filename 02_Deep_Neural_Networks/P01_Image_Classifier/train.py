#!coding: utf-8
"""
Udacity - Airbus Artificial Intelligence Nanodegree
Cédric Campguilhem - Mars 2020
"""
# Future imports
from __future__ import annotations

# Standard library
import argparse
import sys
from typing import Sequence, Tuple, Optional
import os

# PyTorch
from torchvision import datasets, transforms
import torch.utils.data

# Project
from model import FlowerClassifier


def cmdline(argv: Sequence[str]) -> None:
    """
    Configure command line interaction

    :param argv: list of arguments passed to the script
    """
    # Configure command line parser
    parser = argparse.ArgumentParser(description="Training a flower classifier.")
    parser.add_argument('data_directory', metavar='PATH', type=str, help='path to data')
    parser.add_argument('--save_dir', metavar='PATH', type=str, help='path to save model checkpoints', default='/tmp')
    parser.add_argument('--arch', metavar='MODEL', type=str, help='pre-trained model', default='VGG16',
                        choices=['VGG16', 'ResNext101-32x8d'])
    parser.add_argument('--hidden_units', metavar='N1[,N2[,N3...]]', type=str,
                        help='number of neurons per hidden layer', default="")
    parser.add_argument('--activation', metavar='F(X)', type=str, help='activation function', default='relu',
                        choices=['relu', 'leaky_relu', 'sigmoid', 'tanh'])
    parser.add_argument('--dropout', metavar='DROPOUT', type=float, help='dropout probability', default=0.2)
    parser.add_argument('--optimizer', metavar='OPTIM', type=str, help='optimization algorithm', default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('--learning_rate', metavar='LR', type=float, help='learning rate', default=0.001)
    parser.add_argument('--epochs', metavar='EPOCHS', type=int, help='number of epochs', default=5)
    parser.add_argument('--gpu', action='store_true', help='run calculations on GPU if possible',
                        default=False, dest='gpu')
    parser.add_argument('--batch_size', metavar='N', type=int, help='batch size', default=32)
    parser.set_defaults(func=train)
    args = parser.parse_args(argv)
    args.func(args)


def train(args: argparse.Namespace) -> None:
    """
    Train neuron network based on inputs provided by user.

    :param args: command line arguments
    """
    # Make sure data directory contains 'train' and 'valid' folders
    print("Checking that data directories exist... ", end="")
    data_dir = args.data_directory
    if not os.path.exists(data_dir):
        print("error")
        raise OSError("The provided path {} has not been found.".format(data_dir))
    for folder in ['train', 'valid']:
        data_subdir = os.path.join(data_dir, folder)
        if not os.path.exists(data_subdir):
            print("error")
            raise OSError("The provided path {} has not been found.".format(data_subdir))
    print("ok")

    # Checking that save path exists
    print("Checking that saving directory exists... ", end="")
    if not os.path.exists(args.save_dir):
        print("error")
        raise OSError(f"The save directory path {args.save_dir} does not exist.")
    print("ok")

    # Processing hidden units
    print("Processing hidden units... ", end="")
    if args.hidden_units:
        hidden_units = list(map(lambda x: int(x.strip()), args.hidden_units.split(",")))
    else:
        hidden_units = []
    print(hidden_units)

    # Configure loaders
    print(f"Configuring data loaders with batch size {args.batch_size}... ", end="")
    train_loader, valid_loader, _, _ = configure_loaders(os.path.join(data_dir, 'train'),
                                                         os.path.join(data_dir, 'valid'),
                                                         batch_size=args.batch_size)
    print("ok")

    # Configuring the neural network
    print("Configuring the neural network... ", end="")
    clf = FlowerClassifier(model=args.arch, layers=hidden_units, activation=args.activation, dropout=args.dropout,
                           optimizer=args.optimizer, learning_rate=args.learning_rate)
    print(dict(clf.get_configuration()))

    # Fitting the model
    if not args.gpu:
        device = "cpu"
    else:
        device = None
    clf.fit(train_loader, valid_loader, device=device, epochs=args.epochs)

    # Saving model check-point
    filename = os.path.join(args.save_dir, 'checkpoint.pth')
    print(f"Saving model check-point to {filename}... ", end="")
    clf.save(filename)
    print("ok")


def configure_loaders(
        train_dir: str,
        valid_dir: str,
        test_dir: Optional[str] = None,
        batch_size: int = 32,
        shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader,
           torch.utils.data.DataLoader,
           Optional[torch.utils.data.DataLoader],
           torch.utils.data.DataLoader]:
    """
    Configure image loaders

    For the training data I am using the following transforms:
    - Images transformations for data augmentation (in order to increase the ability of neural network to generalize)
        - a random rotation of 30° (around the center of the image)
        - a random horizontal flip (with 50% probability)
        - a random vertical flip (with 50% probability)
        - a random resized crop (as the result the iamge will be 224x224 pixels): this is also required because images
        must be 224x224 to fit into the input layer of the neural network
    - Tensor transformations:
        - a normalization with provided means [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]
        respectively for each color channel (red, green, blue)

    For the testing and validating, I do not use the data augmentation transformations. The RandomResizedCrop is
    replaced with a Resize so that images have the expected size in pixels. The same normalization of tensor is applied
    though.

    The batch size is set to 32 by default so that the neural network may fit into the memory of my GPU device.

    Notes:

    The `dummy_train_dataset` and associated `dummy_train_loader` are only used to illustrate the differences between
    original images and the ones obtained after data augmentation.

    :param train_dir: path to training data
    :param valid_dir: path to validation data
    :param test_dir: path to testing data
    :param batch_size: batch size
    :param shuffle: toggle for train loader shuffling
    :return: image loaders (train, valid, test, dummy)
    """
    # Configure transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Generate datasets
    dummy_train_dataset = datasets.ImageFolder(train_dir, transform=test_transforms)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    if test_dir is not None:
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    else:
        test_dataset = None
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Generate loaders
    dummy_train_loader = torch.utils.data.DataLoader(dummy_train_dataset, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    else:
        test_loader = None
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # End
    return train_loader, valid_loader, test_loader, dummy_train_loader


if __name__ == "__main__":
    cmdline(sys.argv[1:])
