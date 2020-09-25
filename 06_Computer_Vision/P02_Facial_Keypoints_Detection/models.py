#!coding: utf-8
"""
Udacity - Airbus Artificial Intelligence Nanodegree
Cédric Campguilhem - Mars 2020
"""
from typing import Optional, Any, Sequence, Tuple, List, Union
from collections import OrderedDict
import traceback
import time
import sys
from functools import reduce
import itertools

import torch
import torch.utils
import torch.nn as nn
from torchvision import models
import numpy as np


class CUDAContext:
    """
    This context captures exceptions. When PyTorch modules are stored in the stack when exceptions are thrown but not
    dealt with, CUDA memory is not emptied.
    """
    def __enter__(self):
        """
        Nothing special append when entering this context.
        """
        pass

    def __exit__(
            self,
            etype: Optional[Exception] = None,
            value: Optional[str] = None,
            tb: Any = None
    ) -> bool:
        """
        Exiting the context.

        Exceptions are not propagated.
        Traceback is printed to default stream.

        :param etype: exception raised
        :param value: exception value
        :param tb: traceback
        :return: bool
        """
        if tb is not None:
            traceback.print_exception(etype, value, tb)
        # That is the return True part that tell Python that exceptions are ignored.
        return True


def output_shape_conv_and_pool_layer(rows: int,
                                     columns: int,
                                     kernel: int,
                                     stride: int = 1,
                                     padding: int = 0,
                                     dilatation: float = 1.) -> Tuple[int, int]:
    """
    Calculate the size of a convolutional or maxpooling layer.

    The formula used comes directly from PyTorch documentation.

    :param rows: number of rows of input
    :param columns: number of columns of inputs
    :param kernel: kernel size of current layer
    :param stride: stride setting of current layer
    :param padding: padding setting of current layer
    :param dilatation: dilatation setting of current layer
    :return: shape of layer
    """
    return (
        int((rows + 2 * padding - dilatation * (kernel - 1) - 1) / stride + 1),
        int((columns + 2 * padding - dilatation * (kernel - 1) - 1) / stride + 1),
    )


def weight_initialization(m: nn.Module) -> None:
    """
    Initialize weight of provided module

    Initialize the weight of linear layers with Kaiming uniform.

    :param m: module
    """
    if type(m) == nn.Linear:
        # nn.init.xavier_uniform_(m.weight, np.sqrt(2.)) # gain is sqrt(2) because we use ReLU
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        # nn.init.uniform(m.weight, 0., 0.)


# noinspection PyUnresolvedReferences
def build_neuron_network(nb_features_map: Union[Sequence[int], None] = None,
                         size_linear_layers: Union[Sequence[int], None] = None,
                         dropout_rate: Union[Tuple[float, float], float] = 0.3,
                         conv_kernel_size: Union[Sequence[int], int] = 3,
                         conv_stride: int = 1,
                         conv_padding: int = 1,
                         conv_activation: str = "relu",
                         conv_architecture: str = "CPD",
                         pool_kernel_size: int = 2,
                         pool_stride: int = 2,
                         dense_activation: str = "relu",
                         pretrained: Union[str, None] = None,
                         grayscale: bool = True,
                         optimizer: str = "Adam",
                         weight_decay: float = 0.,
                         learning_rate: float = 0.001,
                         ) -> Tuple[nn.Module, List, torch.optim.Optimizer]:
    """
    This function creates a CNN model based on the provided hyper-parameters.
    This will be useful to make hyper-parameter optimization later.

    The CNN basic structure is inspired from the following article:

    https://arxiv.org/pdf/1710.00977.pdf

    And can be summarized this way:

    - feature extractor: n x (Convolution + Activation + Max-pooling + Dropout) + Flatten
    - regressor: m x (layers composed of Linear + Activation + Dropout) + Linear

    It is also possible to specify a different architecture for the feature extractor. The one above is "CPD" meaning
    that a Pool layer is added after each Convolution layer and is then followed by dropout. "CCP" would mean that two
    conv layers are chained before pooling is done. For example a VGG would be "CCPCCPCCCPCCCPCCCP". If the
    architecture is set to "CP" but you provide multiple layers in `nb_features_map`, then the architecture is cycled.

    If you want to use batch normalization instead of dropout in feature extractor use B letter instead of D.

    The function also creates the optimizer algorithm to be used for fitting the model.

    :param nb_features_map: number of feature map, the length of the sequence configures the number of layers in the
    extractor
    :param size_linear_layers: number of neurons in one Linear layer, the length of the sequence configures the number
    of layers in the regressor.
    :param dropout_rate: The first item is probability of dropping out neuron in the network, the second is the
    increment of probability applied to subsequent dropout layers in the network. If only one value is provided the
    increment is set to 0.
    :param conv_kernel_size: kernel size of convolutional units
    :param conv_stride: stride parameter for convolutional units
    :param conv_padding: number of pixels around the image for zero padding
    :param conv_activation: activation function to be used for convolutional layers
    :param conv_architecture: architecture for the convolutional network (feature extractor)
    :param pool_kernel_size: kernel size of pooling units
    :param pool_stride: stride parameter for pooling units
    :param dense_activation: activation function to be used for dense layers
    :param pretrained: use a pretrained model
    :param grayscale: toggle to use grayscale 1-channel or 3-channel images
    :param optimizer: optimizer algorithm to be used
    :param weight_decay: weight decay regularization for optimizer
    :param learning_rate: learning rate of optimizer algorithm
    :return: feature extractor and regressor modules, shapes of different layers, optimizer
    """
    # Initializations
    if pretrained is not None:
        grayscale = False
    if grayscale:
        channels = 1
    else:
        channels = 3
    if nb_features_map is None:
        nb_features_map = [8]
    if size_linear_layers is None:
        size_linear_layers = []
    height = 224
    width = 224
    module = nn.Module()
    shapes = [("input", channels, height, width)]
    layers = {"extractor": [], "regressor": []}
    if not hasattr(dropout_rate, "__len__"):
        dropout_rate = (dropout_rate, 0.)
    next_dropout_rate = dropout_rate[0]
    # If a pretrained model is used:
    if pretrained is None:
        # Input checks
        if hasattr(conv_kernel_size, "__len__"):
            if len(conv_kernel_size) != len(nb_features_map):
                raise ValueError("The length of nb_features_map shall match the length of conv_kernel_size")
        else:
            conv_kernel_size = [conv_kernel_size] * len(nb_features_map)
        # Feature extractor
        next_layer_type = itertools.cycle(conv_architecture)
        nb_feature_map = None
        i = 0
        while True:
            layer_type = next(next_layer_type)
            if layer_type == "C":
                # Convolutional layer
                try:
                    nb_feature_map = nb_features_map[i]
                except IndexError:
                    break
                name = "conv2d-{:02d}".format(i+1)
                conv = nn.Conv2d(shapes[-1][1], nb_feature_map, conv_kernel_size[i], stride=conv_stride,
                                 padding=conv_padding)
                layers["extractor"].append((name, conv))
                h, w = output_shape_conv_and_pool_layer(rows=shapes[-1][2], columns=shapes[-1][3],
                                                        kernel=conv_kernel_size[i], stride=conv_stride,
                                                        padding=conv_padding)
                shapes.append((name, nb_feature_map, h, w))
                i += 1
                # Activation
                if conv_activation == "relu":
                    activ = nn.ReLU()
                elif conv_activation == "elu":
                    activ = nn.ELU(alpha=0.1)
                elif conv_activation == "leaky":
                    activ = nn.LeakyReLU()
                else:
                    activ = nn.ReLU()
                name = "{}-{:02d}".format(conv_activation, i)
                layers["extractor"].append((name, activ))
                # activation does not change the size
                shapes.append((name, shapes[-1][1], shapes[-1][2], shapes[-1][3]))
            elif layer_type == "P":
                # Max-pooling
                name = "maxpool2d-{:02d}".format(i)
                pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
                layers["extractor"].append((name, pool))
                h, w = output_shape_conv_and_pool_layer(rows=shapes[-1][2], columns=shapes[-1][3],
                                                        kernel=pool_kernel_size, stride=pool_stride)
                shapes.append((name, nb_feature_map, h, w))
            elif layer_type == "D":
                # Dropout
                if next_dropout_rate > 0.:
                    name = "dropout-{:02d}".format(i)
                    dropout = nn.Dropout(p=next_dropout_rate)
                    layers["extractor"].append((name, dropout))
                    # Dropout does not change the size
                    shapes.append((name, shapes[-1][1], shapes[-1][2], shapes[-1][3]))
                    next_dropout_rate += dropout_rate[1]
            elif layer_type == "B":
                # Batch normalization
                name = "batchnorm-{:02d}".format(i)
                batch = nn.BatchNorm2d(shapes[-1][1])
                layers["extractor"].append((name, batch))
                # Batch norm. does not change the size
                shapes.append((name, shapes[-1][1], shapes[-1][2], shapes[-1][3]))
        # Add a flatten layer
        name = "flatten"
        flatten = nn.Flatten(1)
        layers["extractor"].append((name, flatten))
        shapes.append((name, shapes[-1][1] * shapes[-1][2] * shapes[-1][3]))
        # Create extractor module
        extractor = nn.Sequential(OrderedDict(layers["extractor"]))
        module.add_module("extractor", extractor)
    elif pretrained == "VGG16":
        pre_trained = models.vgg16(pretrained=True)
        modules = []
        for _name, _module in pre_trained.named_children():
            if _name != 'classifier':
                modules.append((_name, _module))
        modules.append(("flatten", nn.Flatten(1)))
        vgg16 = nn.Sequential(OrderedDict(modules))
        # Freeze all parameters in the pre-trained model
        # So we prevent gradients from being calculated, it will save computation time
        for param in vgg16.parameters():
            param.requires_grad = False
        module.add_module('extractor', vgg16)
        shapes.append((pretrained, 25088))
    else:
        raise ValueError(f"Unknown pre-trained model '{pretrained}'.")
    # Regressor
    for i, size_linear_layer in enumerate(size_linear_layers):
        # Add a linear layer
        name = "linear-{:02d}".format(i + 1)
        linear = nn.Linear(shapes[-1][1], size_linear_layer)
        layers["regressor"].append((name, linear))
        shapes.append((name, size_linear_layer))
        # Activation
        if dense_activation == "relu":
            activ = nn.ReLU()
        elif dense_activation == "elu":
            activ = nn.ELU(alpha=0.1)
        elif dense_activation == "leaky":
            activ = nn.LeakyReLU()
        else:
            activ = nn.ReLU()
        name = "{}-{:02d}".format(dense_activation, i + 1)
        layers["regressor"].append((name, activ))
        shapes.append((name, shapes[-1][1]))  # activation does not change the size
        # Dropout
        if next_dropout_rate > 0.:
            name = "dropout-{:02d}".format(i + 1)
            dropout = nn.Dropout(p=next_dropout_rate)
            layers["regressor"].append((name, dropout))
            shapes.append((name, shapes[-1][1]))  # Dropout does not change the size of array
            next_dropout_rate += dropout_rate[1]
    # Add the final layer, the output size is fixed to 68 x 2 = 136
    name = "output"
    linear = nn.Linear(shapes[-1][1], 136)
    layers["regressor"].append((name, linear))
    shapes.append((name, 136))
    # Create regressor module
    regressor = nn.Sequential(OrderedDict(layers["regressor"]))
    module.add_module("regressor", regressor)
    # Weight initialization
    module.apply(weight_initialization)
    # Optimizer
    if optimizer == "Adam":
        optim = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optim = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optim = torch.optim.SGD(module.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}.")
    return module, shapes, optim


class Net(nn.Module):
    """
    Convolutional Neural Network for facial key points regression.
    """
    def __init__(self, **kwargs):
        """
        Constructor

        Arguments of constructor are passed to the function `build_neuron_network`.
        """
        #super(Net, self).__init__()
        nn.Module.__init__(self)
        # Build CNN
        module, shapes, optim = build_neuron_network(**kwargs)
        self._configuration = kwargs
        self.add_module('cnn', module)
        self.shapes = shapes
        # Loss and optimization
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim
        self._kwargs = kwargs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in the CNN

        :param x: batch of images to be processed
        """
        x = self.cnn.extractor.forward(x)
        return self.cnn.regressor(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict key points on given images

        :param x: batch of images to be processed
        :return: key points coordinates
        """
        # Set model to evaluation mode to deactivate dropouts
        self.eval()
        # Forward pass in the network
        return self.forward(x)

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            epochs: int,
            device: Optional[str] = None,
    ) -> Sequence[Sequence[float]]:
        """
        Fit the model using data from the train_loader.

        Return 3 lists with evolution of metrics all along epochs.
        - losses on the training set
        - elapsed time

        By default, the fitting is done on GPU if available otherwise CPU is used instead. It is possible to force
        cpu by specifying 'cpu' to device parameter.

        The function is protected with a context manager as this prevents, in the case of an exception, to hold
        unnecessary references to the model in the stack. This causes CUDA memory not to be released which results
        in failure in all subsequent attempts to fit any model on GPU device.

        :param train_loader: train loader
        :param valid_loader: validation loader
        :param epochs: number of epochs
        :param device: device to be used for fitting: 'cuda:0' or 'cpu'
        """
        with CUDAContext():
            # Select the device for computation (either GPU or CPU)
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)
            print(f"Fitting the model using {device} as device.")

            # Send the model to the selected device
            self.to(device)

            # Set model to train mode (activates the dropouts if any)
            self.train()

            # Loop over epochs
            start_time = time.time()
            train_losses, valid_losses, elapsed_times = [], [], []
            total_steps_train = 0
            for epoch in range(epochs):
                # Initialization
                running_loss = 0.
                steps_train = 0

                # Loop over batches of images:
                for ii, data in enumerate(train_loader):
                    # Get the input images and their corresponding targets
                    images = data['image']
                    true_key_points = data['keypoints']
                    steps_train += images.shape[0]
                    total_steps_train += images.shape[0]

                    # Flatten pts
                    true_key_points = true_key_points.view(true_key_points.size(0), -1)

                    # Convert variables to floats for regression loss
                    true_key_points = true_key_points.type(torch.FloatTensor)
                    images = images.type(torch.FloatTensor)

                    # Move input and label tensors to the selected device
                    images, true_key_points = images.to(device), true_key_points.to(device)

                    # Reset the accumulated gradients on a previous batch
                    self.optimizer.zero_grad()

                    # Forward pass of neural network (this calculates the facial key points)
                    pred_key_points = self.forward(images)

                    # Calculate the loss and back-propagate the loss in the network to calculate gradients
                    loss = self.criterion(pred_key_points, true_key_points)
                    loss.backward()

                    # Update weights and bias with optimizer
                    self.optimizer.step()

                    # Progress
                    running_loss += (loss.item() * pred_key_points.shape[0] * pred_key_points.shape[1])
                    current_time = time.time()
                    sys.stdout.write(
                        "\rEpoch completion: {:.1f}% Images per second: {:.1f} "
                        "Running loss: {:.3f} "
                        "Elapsed time: {:.1f} seconds ".format(
                            (ii + 1) / len(train_loader) * 100,
                            total_steps_train / (current_time - start_time),
                            running_loss / float(steps_train * pred_key_points.shape[1]),
                            (current_time - start_time)
                        )
                    )
                else:
                    # At the end of each epoch
                    sys.stdout.write("\n")
                    train_losses.append(running_loss / float(steps_train * pred_key_points.shape[1]))
                    elapsed_times.append((current_time - start_time))

                    # Process with a validation step
                    valid_loss = 0.
                    steps_valid = 0

                    # Toggle the model to evaluation mode (it deactivates the dropout)
                    self.eval()
                    with torch.no_grad():
                        # We deactivate gradient computation
                        for ii, data in enumerate(valid_loader):
                            # Get image and key points
                            images = data['image']
                            true_key_points = data['keypoints']
                            steps_valid += images.shape[0]

                            # Flatten pts
                            true_key_points = true_key_points.view(true_key_points.size(0), -1)

                            # Convert variables to floats for regression loss
                            true_key_points = true_key_points.type(torch.FloatTensor)
                            images = images.type(torch.FloatTensor)

                            # Move input and label tensors to the selected device
                            images, true_key_points = images.to(device), true_key_points.to(device)

                            # Forward pass of neural network (this calculates the facial key points)
                            pred_key_points = self.forward(images)

                            # Calculate the loss
                            loss = self.criterion(pred_key_points, true_key_points)

                            # Progress
                            valid_loss += (loss.item() * pred_key_points.shape[0] * pred_key_points.shape[1])

                    # Toggle model back to train mode (it reactivates the dropout)
                    self.train()

                    # Collect losses
                    valid_losses.append(valid_loss / float(steps_valid * pred_key_points.shape[1]))
                    current_time = time.time()

                    # Display status
                    print("Epoch: {}/{}".format(epoch + 1, epochs),
                          "Training loss: {:.3f}".format(train_losses[-1]),
                          "Valid loss: {:.3f}".format(valid_losses[-1]))

        # When CUDA context is terminated
        self.to('cpu')
        return train_losses, valid_losses, elapsed_times

    def save(self, dest: str) -> None:
        """
        Save the model into a checkpoint.

        :param dest: path of the checkpoint to be created.
        """
        # Get the state dictionary
        model_state = self.state_dict()

        # Add some information for our specific module:
        model_state['additional_state'] = {}
        model_state['additional_state']['configuration'] = self._configuration

        # Serialize model
        torch.save(model_state, dest)

    @classmethod
    def load(cls, src: str):
        """
        Load model from serialization file (check-point)

        :param src: path to check-point
        :return: facial keypoint detection module
        """
        # Load check-point
        state_dict = torch.load(src)

        # Create a new module
        specifics = state_dict.pop("additional_state")
        configuration = specifics["configuration"]
        module = Net(**configuration)

        # Restore state
        module.load_state_dict(state_dict)

        # End
        return module

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count the number of parameters in the network. Both the number of trained parameters and overall parameters is
        returned.

        :return: (number of training parameters, total number of parameters)
        """
        c_trained, c_total = 0, 0
        for p in self.parameters():
            increment = reduce(lambda x, y: x * y, p.size())
            if p.requires_grad:
                c_trained += increment
            c_total += increment
        return c_trained, c_total

    def sub_model(self, layer: str) -> nn.Module:
        """
        Extract a sub model up to specified layer

        :param layer: layer name
        :return: sub-model
        """
        submodel = OrderedDict()
        for name, module in self.named_modules():
            children = [(n, m) for n, m in module.named_children()]
            if children:
                continue
            submodel[name.split('.')[-1]] = module
            if name == layer:
                break
        return nn.Sequential(submodel)
