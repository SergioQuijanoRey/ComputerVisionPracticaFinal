import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

import board
import filesystem
from train_loggers import TrainLogger, SilentLogger

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# TODO -- use metrics module to calculate this metrics
def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, classes: list):
    """
    Loads a neural net to make predictions

    Parameters:
    ===========
    model: model that we want to test
    test_loader: pytorch data loader wrapping test data
    classes: list having the names of the different classes we're working with
    """

    # Get current device
    device = get_device()

    # Move to GPU if needed
    model.to(device)

    # Make predictions over test dataset
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # Calculate accuracy of the model
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:

            # Unpack a test data minibatch
            # Then, calculate outputs
            images, labels = unwrap_data(data)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on {len(test_loader)} test images: %d %%' % (
        100 * correct / total))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:

            # Unpack minibatch and calculate outputs
            images, labels = unwrap_data(data)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))

def split_train_test(dataset, train_percentage: float = 0.8):
    """
    Splits a pytorch dataset into train / test datasets

    Parameters:
    ===========
    dataset: the pytorch dataset
    train_percentage: percentage of the dataset given to train

    Returns:
    ========
    train_dataset
    test_dataset
    """

    # Calculate sizes
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_percentage)
    test_size = dataset_size - train_size

    # Split and return
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_device() -> str:
    """
    Returns the most optimal device available
    ie. if gpu available, return gpu. Else return cpu

    Returns:
    ========
    device: string representing the device
    """

    if torch.cuda.is_available():
      device = "cuda:0"
    else:
      device = "cpu"
    return device

def get_datetime_str() -> str:
    """Returns a string having a date/time stamp"""
    return datetime.now().strftime("%d-%m-%Y--%H:%M:%S")


def train_model(
    net: nn.Module,
    path: str,
    parameters: dict,
    train_loader: DataLoader,
    validation_loader: DataLoader = None,
    name: str = "Model",
    logger: TrainLogger = None,
    snapshot_iterations: int = None
) -> None:
    """
    Trains and saves a neural net

    Parameters:
    ===========
    net: Module representing a neural net to train
    path: dir where models are going to be saved
    parameters: dict having the following data:
                - "lr": learning rate
                - "momentum": momentum of the optimizer
                - "criterion": loss function
                - "epochs": epochs to train
    train_loader: pytorch DataLoader wrapping training set
    validation_loader: pytorch DataLoader wrapping validation set
    name: name of the model, in order to save it
    train_logger: to log data about trainning process
                  Default logger is silent logger
    snapshot_iterations: at how many iterations we want to take an snapshot of the model
                         If its None, no snapshots are taken
    """

    # Loss and optimizer
    lr = parameters["lr"]
    momentum = parameters["momentum"]
    criterion = parameters["criterion"]
    
    # TODO -- usar ADAM en vez de SGD
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum = momentum)

    # Select proper device and move the net to that device
    device = get_device()
    net.to(device)

    # Check if no logger is given
    if logger is None:
        print("==> No logger given, using Silent Logger")
        logger = SilentLogger()

    # Printing where we're training
    print(f"==> Training on device {device}")
    print("")

    # Training the network
    epochs = parameters["epochs"]
    for epoch in range(epochs):

        for i, data in enumerate(train_loader):
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = [net(item[None, ...].to(device)) for item in data]
            loss = criterion(outputs)

            # Backward + Optimize
            loss.backward()
            optimizer.step()

            # Statistics -- important, we have to use the iteration given by current epoch and current
            # iteration counter in inner loop. Otherwise logs are going to be non-uniform over iterations
            curr_it = epoch * len(train_loader.dataset) + i * train_loader.batch_size
            if logger.should_log(curr_it):
                logger.log_process(train_loader, validation_loader, epoch, i)

            # Snapshots -- same as Statistics, we reuse current iteration calc
            # TODO -- create a SnapshotTaker class as we have for logs -- snapshot_taker.should_log(i)
            if snapshot_iterations is not None and curr_it % snapshot_iterations == 0:
                # We take the snapshot
                snapshot_name = "snapshot_" + name + "==" + get_datetime_str()
                snapshot_folder = os.path.join(path, "snapshots")
                filesystem.save_model(net, folder_path = snapshot_folder, file_name = snapshot_name)

    print("Finished training")

    # Save the model -- use name + date stamp to save the model
    date = get_datetime_str()
    name = name + "==" + date
    filesystem.save_model(net = net, folder_path = path, file_name = name)

