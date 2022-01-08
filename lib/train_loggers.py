from abc import ABC, abstractmethod # Abstrac classes

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

import core
import board
import metrics

class TrainLogger(ABC):
    """TrainLogger logs data from the trainning process"""

    @abstractmethod
    def log_process(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, iteration: int) -> None:
        """
        Logs an iteration of training process. This log can be just printing to terminal or saving
        scalars to a tensorboard

        Parameters:
        ============
        train_loader: dataloader for training data
        validation_loader: dataloader for validation data
        loss: computed loss in training function
        epoch, iteration: in order to have more logging capabilities
        """
        pass

    @abstractmethod
    def should_log(self, iteration: int) -> bool:
        """Decides wether or not we should log data in this training iteration"""
        pass


class ClassificationLogger(TrainLogger):
    """
    Logger for a classifaction problem

    """
    def __init__(self, net: nn.Module, iterations, loss_func, training_perc: float = 1.0, validation_perc: float = 1.0):
        """
        Initializes the logger

        Parameters:
        ===========
        net: the net we are testing
        iterations: how many iterations we have to wait to log again
        loss_func: the loss func we are using to train
        training_perc: the percentage of the training set we are going to use to compute metrics
        validation_perc: the percentage of the validation set we are going to use to compute metrics
        tensorboardwriter: writer to tensorboard logs
        """
        self.iterations = iterations
        self.loss_func = loss_func
        self.net = net
        self.training_perc = training_perc
        self.validation_perc = validation_perc

        # Tensorboard named with the date/time stamp
        self.name = core.get_datetime_str()
        self.tensorboardwriter = board.get_writer(name = self.name)

    def log_process(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, iteration: int) -> None:

        # For more performance
        with torch.no_grad():

            # Even more performance
            self.net.eval()

            # Trainning loss and accuracy
            max_examples = int(len(train_loader.dataset) * self.training_perc)
            mean_train_loss = metrics.calculate_mean_loss(self.net, train_loader, max_examples, self.loss_func)
            mean_train_acc = metrics.calculate_accuracy(self.net, train_loader, max_examples)

            # Validation loss and accuracy
            max_examples = int(len(validation_loader.dataset) * self.training_perc)
            mean_val_loss = metrics.calculate_mean_loss(self.net, validation_loader, max_examples, self.loss_func)
            mean_val_acc = metrics.calculate_accuracy(self.net, validation_loader, max_examples)



        # Set again net to training mode
        self.net.train()

        # Output to the user
        # We don't care about epochs starting in 0, but with iterations is weird
        # ie. epoch 0 it 199 instead of epoch 0 it 200
        print(f"[{epoch} / {iteration}]")
        print(f"Training loss: {mean_train_loss}")
        print(f"Validation loss: {mean_val_loss}")
        print(f"Training acc: {mean_train_acc}")
        print(f"Validation acc: {mean_val_acc}")
        print("")

        # Sending this metrics to tensorboard
        curr_it = iteration * train_loader.batch_size + epoch * len(train_loader.dataset) # The current iteration taking in count
                                                                # that we reset iterations at the end
                                                                # of each epoch

        # Send data to tensorboard
        # We use side-by-side training / validation graphics
        self.tensorboardwriter.add_scalars(
            "Loss",
            {
                "Training loss": mean_train_loss,
                "Validation loss": mean_val_loss,
            },
            curr_it
        )
        self.tensorboardwriter.add_scalars(# Have train / val acc in same graph to compare
            "Accuracy",
            {
                "Training acc": mean_train_acc,
                "Validation acc": mean_val_acc,
            },
            curr_it
        )
        self.tensorboardwriter.flush() # Make sure that writer writes to disk

    def should_log(self, iteration: int) -> bool:
        if iteration % self.iterations == 0:
            return True

        return False

class SilentLogger(TrainLogger):
    """Logger that does not log data"""

    def log_process(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, iteration: int) -> None:
        pass

    def should_log(self, iteration: int) -> bool:
        # Always return false in order to never log
        return False

class TripletLogger(TrainLogger):
    """
    Custom logger for triplet training
    """

    def __init__(self, net: nn.Module, iterations, loss_func, training_perc: float = 1.0, validation_perc: float = 1.0):
        """
        Initializes the logger

        Parameters:
        ===========
        net: the net we are testing
        iterations: how many iterations we have to wait to log again
        loss_func: the loss func we are using to train <- Should be some triplet-like loss
        training_perc: the percentage of the training set we are going to use to compute metrics
        validation_perc: the percentage of the validation set we are going to use to compute metrics
        tensorboardwriter: writer to tensorboard logs
        """
        self.iterations = iterations
        self.loss_func = loss_func
        self.net = net
        self.training_perc = training_perc
        self.validation_perc = validation_perc

        # Tensorboard named with the date/time stamp
        self.name = core.get_datetime_str()
        self.tensorboardwriter = board.get_writer(name = self.name)

    def log_process(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, iteration: int) -> None:
        print(f"Entrenando epoca {epoch} iteracion {iteration}")

    def should_log(self, iteration: int) -> bool:
        if iteration % self.iterations == 0:
            return True

        return False

