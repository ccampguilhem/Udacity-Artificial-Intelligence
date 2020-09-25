#!coding: utf-8
"""
Udacity - Airbus Artificial Intelligence Nanodegree
Cédric Campguilhem - Mars 2020
"""
from typing import Optional, Mapping, Callable, Union, Tuple
import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.image
import pandas as pd
import cv2

# Type aliases
Sample = Mapping[str, Union[np.ndarray, torch.Tensor]]
Transform = Callable[[Sample], Sample]


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,
                 csv_file: str,
                 root_dir: str,
                 transform: Optional[Transform] = None):
        """
        Constructor

        :param csv_file: Path to the csv file with annotations.
        :param root_dir: Directory with all the images.
        :param transform: Optional transform to be applied on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Return number of sample in dataset.

        :return: length of dataset
        """
        return len(self.key_pts_frame)

    def __getitem__(self, idx: int) -> Sample:
        """
        Return sample at specified index.

        :param idx: sample index to be retrieved
        :return:
        """
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = matplotlib.image.imread(image_name)
        # If image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalize:
    """
    Convert a color image to grayscale and normalize the color range to [0,1].
    """
    def __init__(self, grayscale: bool = True):
        """
        Constructor

        :param grayscale: convert to gray scale
        """
        self._grayscale = grayscale

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply transformation on provided sample.

        :param sample:
        :return: transformed sample
        """
        image, key_pts = sample['image'], sample['keypoints']
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        # Convert image to grayscale
        if self._grayscale:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        # Scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0
        # Scale keypoints to be centered around 0 with a range of [-1, 1]
        # Mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100) / 50.0
        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale:
    """
    Rescale the image in a sample to a given size.
    """
    def __init__(self, output_size: Union[Tuple[int], int]):
        """
        Constructor

        :param output_size: Desired output size. If tuple, output is matched to output_size. If int, smaller of image
        edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply transformation on provided sample.

        :param sample:
        :return: transformed sample
        """
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]
        return {'image': img, 'keypoints': key_pts}


class RandomCrop:
    """
    Crop randomly the image in a sample.
    """
    def __init__(self, output_size: Union[Tuple[int], int]):
        """
        Constructor

        :param output_size: Desired output size. If int, square crop is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply transformation on provided sample.

        :param sample:
        :return: transformed sample
        """
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        key_pts = key_pts - [left, top]
        return {'image': image, 'keypoints': key_pts}


class RandomRotate:
    """
    Randomly rotate an image.
    """

    def __init__(self, max_angle: float = 30.):
        """
        Constructor

        :param max_angle: image will be rotated with an angle within [-max_angle, +max_angle]
        """
        self.max_angle = max_angle

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply transformation on provided sample.

        :param sample:
        :return: transformed sample
        """
        # Get rotation angle, image and key points
        angle = np.random.uniform(-self.max_angle, self.max_angle, 1)[0]
        image = sample["image"]
        key_points = sample["keypoints"]
        # Calculate rotation matrix
        image_center = image.shape[1] // 2, image.shape[0] // 2
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # Rotate image
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        # Rotate coordinates
        key_points = np.hstack((key_points, np.ones((key_points.shape[0], 1))))
        key_points = np.dot(rot_mat, key_points.T).T
        # End
        return {"image": image, "keypoints": key_points}


class RandomFlip:
    """
    Randomly vertically flip an image.
    """

    def __init__(self, probability: float = 0.3):
        """
        Constructor

        :param probability: probability of the image to be flipped
        """
        self.probability = probability

    def __call__(self, sample: Sample) -> Sample:
        """
        Apply transformation on provided sample.

        :param sample:
        :return: transformed sample
        """
        # State whether we flip the image:
        if random.uniform(0., 1.) <= self.probability:
            image = sample["image"]
            key_points = sample["keypoints"].copy()
            w = image.shape[1]
            key_points[:, 0] = w - key_points[:, 0]
            image = image.copy()
            image = cv2.flip(image, 1)
            sample = {"image": image, "keypoints": key_points}
        return sample


class ToTensor:
    """
    Convert numpy arrays in sample to Tensors.
    """
    def __call__(self, sample: Sample) -> Sample:
        """
        Apply transformation on provided sample.

        :param sample:
        :return: transformed sample
        """
        image, key_pts = sample['image'], sample['keypoints']
        # If image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # Add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # Swap color axis because
        # Numpy image: H x W x C
        # Torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(key_pts)}
