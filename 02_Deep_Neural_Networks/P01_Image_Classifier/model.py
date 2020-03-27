#!coding: utf-8
"""
Udactity - Airbus Artificial Intelligence Nanodegree
Cédric Campguilhem - Mars 2020
"""
# Future imports
from __future__ import annotations

# Standard library
import sys
import time
import traceback
import json
import os
from typing import Optional, Sequence, Mapping, Tuple, Any
from collections import OrderedDict

# PyTorch
import torch
from torch import nn
from torch import optim
import torch.utils.data
# import torch.nn.functional as F
from torchvision import models

# Other
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import numpy as np
import PIL.Image


def imshow(
        image: torch.Tensor,
        ax: Optional[matplotlib.axes.Axes] = None,
        normalize: bool = True
) -> matplotlib.axes.Axes:
    """
    Show tensor as an image.

    :param image: tensor containing image
    :param ax: subplot to draw image
    :param normalize: specify whether input image has been normalized
    :return: matplotlib subplot
    """
    # Create subplot if none is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Convert tensor into image for imshow
    image = image.numpy().transpose((1, 2, 0))

    # Apply normalization backwards
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    # Plot the image and configure the subplot
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    # End
    return ax


def show_six(loader: torch.utils.data.DataLoader) -> None:
    """
    Display 6 images from the given data loader

    :param loader: torchvision data loader
    :return: None
    """
    # Configure loader
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Create figure and subplots
    fig, axes = plt.subplots(figsize=(15, 4), ncols=6)
    for ii in range(6):
        ax = axes[ii]
        imshow(images[ii], ax=ax, normalize=True)


def category_plot(prediction: Mapping[str, float], tensor: torch.Tensor) -> matplotlib.figure.Figure:
    """
    
    :param prediction: 
    :param tensor: 
    :return: 
    """
    # Create plot layout
    fig, axes = plt.subplots(figsize=(8, 3), ncols=2)

    # Display a picture of the flower
    imshow(tensor, axes[1])
    axes[1].axis('off')

    # Display a horizontal bar plot
    classes = list(prediction.keys())
    classes.reverse()
    probabilities = list(prediction.values())
    probabilities.reverse()
    axes[0].barh(classes, probabilities)

    # Set title
    fig.suptitle(classes[-1])

    # Finalization
    plt.subplots_adjust(top=0.80)
    plt.tight_layout(pad=2.0)
    return fig


def load_categories(filename: str) -> Mapping[int, str]:
    """
    Load categories from specified file.

    :param filename: path to category filename
    :return: a dictionary of flowers id and name
    """
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


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


def build_neuron_network(
        layers: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
        activation: str = "relu",
        input_layer: int = 25088,
        output_layer: int = 102
) -> nn.Module:
    """
    Build the classifier neural network

    :param layers: internal layers to be added to the network
    :param dropout: dropout probability to be added to the layers
    :param activation: activation function to be used: can be "relu", "leaky_relu", "tanh" or "sigmoid"
    :param input_layer: number of neurons in the input layer
    :param output_layer: number of neurons in the output layer
    :return: PyTorch module
    """
    # Set activation function
    activation_func = None
    if activation == "relu":
        activation_func = nn.ReLU()
    elif activation == "tanh":
        activation_func = nn.Tanh()
    elif activation == "sigmoid":
        activation_func = nn.Sigmoid()
    elif activation == "leaky_relu":
        activation_func = nn.LeakyReLU()
    elif activation is not None:
        raise ValueError(f"Unknown activation function: {activation}")

    # Set size of layers
    layer_sizes = [input_layer]
    if layers is None:
        layer_sizes.append(output_layer)
    else:
        for layer in layers:
            layer_sizes.append(layer)
        layer_sizes.append(output_layer)

    # Stack layers
    layers = []
    for i in range(len(layer_sizes) - 1):
        last_layer = (i == len(layer_sizes) - 2)
        if not last_layer:
            layers.append((f'layer_{i + 1}', nn.Linear(layer_sizes[i], layer_sizes[i + 1])))
            if activation is not None:
                layers.append((activation, activation_func))
            if dropout > 0.:
                layers.append((f'dropout_{i + 1}', nn.Dropout(p=dropout)))
        else:
            layers.append(('output', nn.Linear(layer_sizes[i], layer_sizes[i + 1])))
            layers.append(('softmax', nn.Softmax(dim=1)))

    # Set classifier
    model = nn.Sequential(OrderedDict(layers))
    return model


class FlowerClassifier(nn.Module):
    """
    PyTorch module for the flower classification project.

    The dropout is used for regularization: at each forward pass, a certain portion of the neural networked is dropout
    and is not used to make the prediction. The intent is to prevent some weights to be too high and force the all
    neurons in the network to be complementary. This shall result in a training accuracy increasing with the validation
    accuracy (on unseen samples) and thus avoid over-fitting.

    The loss function used is `Negative Log Likelihood <https://pytorch.org/docs/stable/nn.html#nllloss>`_ function.
    This function expects to have log softmax as input while our model only provides softmax. This means that we have
    to calculates the log of model predictions before feeding the NLLLoss function. That way we are implementing a
    cross-entropy loss function like `this <https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_.
    It is better not to use probabilities in the optimization process itself. This way the model `forward` method can
    be used to classify images once the model has been trained.
    """
    def __init__(
            self,
            model: str = "VGG16",
            layers: Optional[Sequence[int]] = None,
            dropout: float = 0.2,
            activation: str = "relu",
            optimizer: str = "adam",
            learning_rate: float = 0.001,
    ):
        """
        Create the neural network architecture and configures the optimizer and loss function.

        :param model: pre-trained model to be used (either VGG16 or ResNext101-32x8d)
        :param layers: size of inner layers in the classifier part of the network
        :param dropout: probability of dropout in the classifier layers
        :param activation: activation function to be used, one of "relu", "leaky_relu", "tanh", "sigmoid"
        :param optimizer: optimization algorithm can be either "adam", "lbfgs" or "sgd"
        :param learning_rate: learning rate for optimizer
        """
        # Inheritance
        super().__init__()

        # Load pre-trained model and freeze its parameters
        input_layer = self._load_pretrained_model(model)

        # Build our own classifier
        classifier = build_neuron_network(layers, dropout, activation, input_layer=input_layer)
        self.add_module('classifier', classifier)

        # Configure optimizer
        self._configure_optimizer(optimizer, learning_rate)

        # Configure the loss function
        self.criterion = nn.NLLLoss()

        # Other initializations
        self._class_to_idx = None
        self._cat_to_name = None
        self.load_categories("cat_to_name.json")

        # Store configuration
        self._configuration = OrderedDict()
        self._configuration["model"] = model
        self._configuration["layers"] = layers
        self._configuration["dropout"] = dropout
        self._configuration["activation"] = activation
        self._configuration["optimizer"] = optimizer
        self._configuration["learning_rate"] = learning_rate

    def _load_pretrained_model(self, model: str) -> int:
        """
        Load pre-trained model and return the required number of input layers for the classifier.

        The parameters of pre-trained model are frozen.

        :param model: pre-trained model to be used (either VGG16 or ResNext101-32x8d)
        :return: required number of neurons for the input layer of classifier
        """
        # Load pre-trained model
        if model == "VGG16":
            pre_trained = models.vgg16(pretrained=True)
            modules = []
            for name, module in pre_trained.named_children():
                if name != 'classifier':
                    modules.append((name, module))
            self.add_module('pretrained', nn.Sequential(OrderedDict(modules)))
            input_layer = 25088
        elif model == "ResNext101-32x8d":
            pre_trained = models.resnext101_32x8d(pretrained=True)
            modules = []
            for name, module in pre_trained.named_children():
                if name != 'fc':
                    modules.append((name, module))
            self.add_module('pretrained', nn.Sequential(OrderedDict(modules)))
            input_layer = 2048
        else:
            raise ValueError("Either 'VGG16' or 'ResNext101-32x8d' has to be used as pre-trained model.")

        # Freeze all parameters in the pre-trained model
        # So we prevent gradients from being calculated, it will save computation time
        for param in self.pretrained.parameters():
            param.requires_grad = False

        # End
        return input_layer

    def _configure_optimizer(self, optimizer: str, learning_rate: float) -> None:
        """
        Configure optimizer

        :param optimizer: optimization algorithm can be either "adam" or "sgd"
        :param learning_rate: learning rate for optimizer
        """
        # Configure the optimizer
        # We only pass the parameters of the classifier module, the pre-trained module is left unchanged
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.classifier.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimization algorithm: {optimizer}")

    def __del__(self):
        """
        This method only enables to check that classifier is garbage-collected properly to avoid memory
        allocation errors in CUDA.
        """
        torch.cuda.empty_cache()
        print("CUDA cache emptied.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Neural network forward pass. Return the probabilities.

        :param x: current batch to be processed
        :return: probabilities
        """
        # Forward pass through the pre-trained model
        x = self.pretrained.forward(x)

        # The tensor is flatten as it is the case in VGG model and in ResNet model
        x = torch.flatten(x, 1)

        # Forward pass through the classifier to be trained
        x = self.classifier.forward(x)
        return x

    def get_configuration(self) -> OrderedDict:
        """
        Return an information related to the configuration of classifier.

        :return: configuration of module
        """
        return self._configuration

    def load_categories(self, path: str) -> None:
        """
        Load categories from provided file path.

        :param path: path to category files (.json)
        """
        try:
            self._cat_to_name = load_categories(path)
        except OSError:
            self._cat_to_name = None

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            epochs: int,
            device: Optional[str] = None,
    ) -> Sequence[Sequence[float]]:
        """
        Fit the model using data from the train_loader and display accuracy with valid_loader.

        Return 4 lists with evolution of metrics all along epochs.
        - losses on the training set
        - losses on the validation set
        - accuracies
        - elapsed time

        By default, the fitting is done on GPU is available otherwise CPU is used instead. It is possible to force
        cpu by specifying 'cpu' to device parameter.

        The function is protected with a context manager as this prevents, in the case of an exception to hold
        unnecessary references to the model in the stack. This causes CUDA memory not to be released which results
        in failure in all subsequent attempts to fit any model on GPU device.

        :param train_loader: train loader
        :param valid_loader: valid_loader to assess accuracy of model
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

            # Store classes to index
            self._class_to_idx = train_loader.dataset.class_to_idx

            # Send the model to the selected device
            self.to(device)

            # Set model to train mode (activates the dropouts if any)
            self.train()

            # Loop over epochs
            start_time = time.time()
            train_losses, valid_losses, accuracies, elapsed_times = [], [], [], []
            steps = 0
            for epoch in range(epochs):
                # Initialization
                running_loss = 0.

                # Loop over batches of images:
                for ii, (images, labels) in enumerate(train_loader):
                    steps += images.shape[0]

                    # Move input and label tensors to the selected device
                    images, labels = images.to(device), labels.to(device)

                    # Reset the accumulated gradients on a previous batch
                    self.optimizer.zero_grad()

                    # Forward pass of neural network (this calculates the probabilities)
                    ps = self.forward(images)

                    # Calculate the log so we can use the loss function
                    log_ps = torch.log(ps)

                    # Calculate the loss and back-propagate the loss in the network to calculate gradients
                    loss = self.criterion(log_ps, labels)
                    loss.backward()

                    # Update weights and bias with optimizer
                    self.optimizer.step()

                    # Progress
                    running_loss += loss.item()
                    current_time = time.time()
                    sys.stdout.write("\rEpoch completion: {:.1f}% Images per second: {:.1f} "
                                     "Elapsed time: {:.1f} seconds".format((ii + 1) / len(train_loader) * 100,
                                                                           steps / (current_time - start_time),
                                                                           (current_time - start_time))
                                     )
                else:
                    # At the end of each epoch, process with a validation step
                    # Initialization
                    sys.stdout.write("\n")
                    valid_loss = 0.
                    accuracy = 0.

                    # Toggle the model to evaluation mode (it deactivates the dropout)
                    self.eval()
                    with torch.no_grad():
                        # We deactivate gradient computation
                        for images, labels in valid_loader:
                            # Move data to selected device
                            images, labels = images.to(device), labels.to(device)

                            # Forward pass
                            ps = self.forward(images)

                            # Loss
                            log_ps = torch.log(ps)
                            loss = self.criterion(log_ps, labels)
                            valid_loss += loss.item()

                            # Keep only the most likely and compare with label
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = (top_class == labels.view(*top_class.shape))
                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    # Toggle model back to train mode (it reactivates the dropout)
                    self.train()

                    # Collect losses
                    accuracy = accuracy.item() / len(valid_loader) * 100.
                    train_losses.append(running_loss / len(train_loader))
                    valid_losses.append(valid_loss / len(valid_loader))
                    accuracies.append(accuracy)
                    current_time = time.time()
                    elapsed_times.append(current_time - start_time)

                    # Display status
                    print("Epoch: {}/{}".format(epoch + 1, epochs),
                          "Training loss: {:.3f}".format(running_loss / len(train_loader)),
                          "Valid loss: {:.3f}".format(valid_loss / len(valid_loader)),
                          "Accuracy: {:.1f}%".format(accuracy))

        # When CUDA context is terminated
        return train_losses, valid_losses, accuracies, elapsed_times

    def test(self, test_loader: torch.utils.data.DataLoader, device: Optional[str] = None) -> Tuple[float, float]:
        """
        Evaluate model on a test dataset.

        By default all calculations are made on GPU if available otherwise on CPU. You can change this behavior by
        specifying the device: 'cuda:0' or 'cpu'.

        :param test_loader: loader for test dataset
        :param device: device to be used for calculations
        :return: tuple with loss and accuracy
        """
        with CUDAContext():
            # Select the device for computation (either GPU or CPU)
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)
            print(f"Testing the model using {device} as device.")

            # Send the model to the selected device
            self.to(device)

            # Set model to eval mode
            self.eval()

            # Initialization
            test_loss = 0.
            accuracy = 0.

            with torch.no_grad():
                # We deactivate gradient computation
                for images, labels in test_loader:
                    # Move data to selected device
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    ps = self.forward(images)

                    # Loss
                    log_ps = torch.log(ps)
                    loss = self.criterion(log_ps, labels)
                    test_loss += loss.item()

                    # Keep only the most likely and compare with label
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            # Collect losses
            accuracy = accuracy.item() / len(test_loader) * 100.
            test_loss = test_loss / len(test_loader)

            # Display status
            print("Testing loss: {:.3f}".format(test_loss),
                  "Accuracy: {:.1f}%".format(accuracy))

        # When CUDA context is terminated
        return test_loss, accuracy

    def process_image(self, path: str) -> Tuple[torch.Tensor, PIL.Image.Image]:
        """
        Convert an image into a tensor

        The image is cropped to a 224x224 size then it is normalized with the following means and standard deviations:
        - mean: [0.485, 0.456, 0.406]
        - standatd deviation [0.229, 0.224, 0.225]

        :param path: path of image to be converted
        :return: tensor
        """
        # Open image from disk
        img = PIL.Image.open(path)
        before = img.copy()

        # Resize to a minimum dimension of 256
        ratio = img.size[0] / img.size[1]
        if img.size[0] > img.size[1]:
            s = int(np.floor(256 * ratio))
        else:
            s = int(np.ceil(256 / ratio))
        img.thumbnail((s, s))

        # Center and crop
        width = img.size[0]
        left = (width - 224) / 2
        right = img.size[0] - (width - 224) / 2
        height = img.size[1]
        upper = (height - 224) / 2
        lower = img.size[1] - (height - 224) / 2
        img = img.crop((left, upper, right, lower))

        # Color channel 0->255 -> 0->1 after conversion to numpy array
        array = np.array(img) / 255.

        # Normalization with mean and standard deviation
        array = (array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # Transposition because PIL images have the color channels in last index while in torchvision there are in
        # first place
        array = array.transpose((2, 0, 1))

        # Convert to PyTorch tensor
        tensor = torch.Tensor(array)

        return tensor, before

    def predict(self, path: str, device: Optional[str] = None, top_k: int = 5) -> Tuple[OrderedDict, torch.Tensor]:
        """
        Return the top-5 probabilities of the provided image

        :param path: path to the image to be processed by the neuron network
        :param device: device to run calculations on 'cpu' or 'cuda:0'
        :param top_k: how many most likely classes are returned
        :return: dictionary of flower name and associated probability and image in a form of torch tensor
        """
        # Convert Image into a tensor
        tensor, _ = self.process_image(path)
        tensor = tensor.view((1, 3, 224, 224))

        with CUDAContext():
            # Select the device for computation (either GPU or CPU)
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)
            print(f"Predicting with the model using {device} as device.")

            # Move tensor to selected device
            tensor.to(device)

            # Toggle model in evaluation mode to de-activate dropouts
            self.eval()

            # Pass the image through the network (de-activate gradients calculations)
            with torch.no_grad():
                ps = self.forward(tensor)

            # Extract the top-5 probabilities
            top_p, top_class = ps.topk(top_k, dim=1)
            top_p = top_p.numpy()
            top_class = top_class.numpy()

        # Index to class
        idx_to_class = {}
        for class_, idx in self._class_to_idx.items():
            idx_to_class[idx] = class_

        # End
        dct = OrderedDict()
        for probability, idx in zip(top_p[0], top_class[0]):
            class_ = idx_to_class[idx]
            if self._cat_to_name is not None:
                name = self._cat_to_name[class_]
            else:
                name = class_
            dct[name] = probability
        return dct, tensor.squeeze()

    def save(self, dest: str) -> None:
        """
        Save the model into a checkpoint.

        :param dest: path of the checkpoint to be created.
        """
        # Get the state dictionary
        model_state = self.state_dict()

        # Add some information for our specific module:
        model_state['FlowerClassifier'] = {}
        model_state['FlowerClassifier']['configuration'] = self._configuration
        model_state['FlowerClassifier']['class_to_idx'] = self._class_to_idx

        # Serialize model
        torch.save(model_state, dest)

    @classmethod
    def load(cls, src: str) -> FlowerClassifier:
        """
        Load model from serialization file (check-point)

        :param src: path to check-point
        :return: flower classifier module
        """
        # Load check-point
        state_dict = torch.load(src)

        # Create a new module
        specifics = state_dict.pop("FlowerClassifier")
        configuration = specifics["configuration"]
        module = FlowerClassifier(**configuration)

        # Restore state
        module._class_to_idx = specifics["class_to_idx"]
        module.load_state_dict(state_dict)

        # End
        return module


def plot_history(
        train_losses: Sequence[float],
        valid_losses: Sequence[float],
        accuracies: Sequence[float],
        conf: str
) -> None:
    """
    Plot convergence history of model for a given configuration

    :param train_losses: loss on training samples
    :param valid_losses: loss on validation samples
    :param accuracies: accuracy on validation samples
    :param conf: model configuration
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = range(1, len(train_losses) + 1)
    lines = []
    line1, = ax.plot(x, train_losses, label="train")
    lines.append(line1)
    line2, = ax.plot(x, valid_losses, label="valid")
    lines.append(line2)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    line3, = ax2.plot(x, accuracies, label="accuracy", color='k', ls='--')
    lines.append(line3)
    ax2.set_ylabel("accuracy (%)")
    ax.set_title(f"{conf}\nAccuracy = {accuracies[-1]:.1f}%")
    ax.legend(lines, [l.get_label() for l in lines], loc='center left', bbox_to_anchor=(1.15, 0.5))


def load_trade_results() -> Sequence[dict]:
    """
    Load metrics from model training.

    The training results contain:
    - model configuration
    - convergence history

    :return: history of training results
    """
    if os.path.exists("results.json"):
        with open("results.json", "r") as fobj:
            data = fobj.read()
            results = json.loads(data)
    else:
        results = []
    return results


def save_trade_results(results: Sequence[dict]) -> None:
    """
    Save history of trainings in a file.

    :param results: history of trainings
    """
    with open("results.json", "w") as fobj:
        fobj.write(json.dumps(results, indent=4))

