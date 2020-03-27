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
from typing import Sequence
import os

# Project
from model import FlowerClassifier


def cmdline(argv: Sequence[str]) -> None:
    """
    Configure command line interaction

    :param argv: list of arguments passed to the script
    """
    # Configure command line parser
    parser = argparse.ArgumentParser(description="Predicting with a flower classifier.")
    parser.add_argument('image', metavar='IMAGE', type=str, help='path to image file')
    parser.add_argument('checkpoint', metavar='CHECKPOINT', type=str, help='path to checkpoint (.pth)')
    parser.add_argument('--category_names', metavar='PATH', type=str, help='path to category file',
                        default='./cat_to_name.json')
    parser.add_argument('--top_k', metavar='K', type=int, help='k-most likely classes', default=5)
    parser.add_argument('--gpu', action='store_true', help='run calculations on GPU if possible',
                        default=False, dest='gpu')
    parser.set_defaults(func=predict)
    args = parser.parse_args(argv)
    args.func(args)


def predict(args: argparse.Namespace) -> None:
    """
    Predict k-most likely classes for provided image.

    :param args: command line arguments
    """
    # Make sure the check-point exists
    print("Checking that check-point exists... ", end="")
    if not os.path.exists(args.checkpoint):
        print("error")
        raise OSError(f"The file {args.checkpoint} has not been found.")
    else:
        print("ok")

    # Make sure the image file exists
    print("Checking that image file exists... ", end="")
    if not os.path.exists(args.image):
        print("error")
        raise OSError(f"The file {args.image} has not been found.")
    else:
        print("ok")

    # Make sure the category file exists
    print("Checking that category file exists... ", end="")
    if not os.path.exists(args.category_names):
        print("error")
        raise OSError(f"The file {args.category_names} has not been found.")
    else:
        print("ok")

    # Loading the check-point
    clf = FlowerClassifier.load(args.checkpoint)
    clf.load_categories(args.category_names)

    # Make prediction
    if not args.gpu:
        device = "cpu"
    else:
        device = None
    classes, _ = clf.predict(args.image, device=device, top_k=args.top_k)
    for i, (class_, probability) in enumerate(classes.items()):
        print(f"{i + 1:02d} - {class_} - {probability * 100.:.1f}%")


if __name__ == "__main__":
    cmdline(sys.argv[1:])
