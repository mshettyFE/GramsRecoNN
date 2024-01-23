import torch
from torch import nn
from torch.utils.data import DataLoader
from AnodePlaneDataset import AnodePlaneDataset
import matplotlib.pyplot as plt

import sys
import argparse

from dataclasses import dataclass, field

sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck


class SimpleNN(nn.Module):
    def __init__(self, PixelCountX, PixelCountY):
        super().__init__()
        self.EscapeStack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride = (2,2)),
            nn.Flatten(),
            nn.Linear(in_features=25, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25,out_features=2),
        )
        self.output_layer = nn.LogSoftmax(dim=2)

    def forward(self,x):
        inner_res = self.EscapeStack(x)
        return self.output_layer(inner_res)        

@dataclass
class History:
    training_loss_hist: list[float] = field(default_factory=list)
    validation_loss_hist: list[float] = field(default_factory=list)
    loss_epoch_count: list[int] = field(default_factory=list)
    training_accuracy_history: list[float] = field(default_factory=list)
    validation_accuracy_history: list[float] = field(default_factory=list)
    acc_epoch_count: list[int] = field(default_factory=list)
    def add_loss(self,training_loss: float, validation_loss:float, epoch_num:int =None):
# Generically adds the training loss and validation_loss to history
# training_loss and validation_loss are just doubles
# Can specify what epoch number these losses correspond to
# If you don't, the epoch number is just the total number of losses recorded (0-based index)
        if epoch_num==None:
            self.loss_epoch_count.append(len(self.training_loss_hist))
        else:
            self.loss_epoch_count.append(epoch_num)
        self.training_loss_hist.append(training_loss)
        self.validation_loss_hist.append(validation_loss)

    def add_accuracy(self, training_percentage:float, validation_percentage:float, epoch_num:int =None):
# Same as loss, but a percentage of how many labels were accurately classified
        if epoch_num==None:
            self.acc_epoch_count.append(len(self.training_accuracy_history))
        else:
            self.acc_epoch_count.append(epoch_num)
        self.training_accuracy_history.append(training_percentage)
        self.validation_accuracy_history.append(validation_percentage)

    def print(self):
        print("Loss")
        print(self.loss_epoch_count)
        print(self.training_loss_hist)
        print(self.validation_loss_hist)
        print("Accuracy")
        print(self.acc_epoch_count)
        print(self.training_accuracy_history)
        print(self.validation_accuracy_history)

    def plot(self, which="loss"):
        match which:
            case "loss":
                pass
            case "acc":
                pass
            case _:
                raise Exception("which must be either 'loss' or 'acc'")

@dataclass
class HyperParameters:
    learning_rate: float
    batch_size: int
    epoch_num: int

class Trainer:
    def __init__(self, parameters,optimizer, loss_func):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.args = HyperParameters(
            float(parameters["TrainData"]["LearningRate"]["value"]),
                                    int(parameters["TrainData"]["BatchSize"]["value"]),
                                    int(parameters["TrainData"]["EpochNum"]["value"])
        )                           
        PixelCountX = int(parameters["GenData"]["PixelCountX"]["value"])
        PixelCountY = int(parameters["GenData"]["PixelCountY"]["value"])
        self.model = SimpleNN(PixelCountX,PixelCountY).to(self.device)
        self.opt = optimizer(self.model.parameters(),  lr=self.args.learning_rate)
        self.loss = loss_func
        self.train_data = DataLoader(AnodePlaneDataset(parameters["TrainData"]["InputTrainingFolderPath"]["value"]),
                                 batch_size=self.args.batch_size, shuffle=True)
        self.val_data = DataLoader(AnodePlaneDataset(parameters["TrainData"]["InputValidationFolderPath"]["value"]),
                              batch_size=self.args.batch_size, shuffle=False)
        self.test_data = DataLoader(AnodePlaneDataset(parameters["TrainData"]["InputTestFolderPath"]["value"]),
                               batch_size=self.args.batch_size, shuffle=False)
        self.training_history = History()

    def evaulate(self, which="validation"):
        match which:
            case "validation":
                pass
            case "test":
                pass
            case _:
                raise Exception("which argument is neither 'validation' nor 'test'")
## For validation set, update history
## For test set, don't update history
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TrainNet')
    parser.add_argument('TOML',help="Path to .toml config file")
    args = parser.parse_args()
    sanity_checker = TomlSanityCheck(args.TOML)
    try:
        sanity_checker.validate()
    except Exception as e:
        print(e)
        sys.exit()
    paras=  sanity_checker.return_config()
    trainer = Trainer(paras, torch.optim.Adam, nn.NLLLoss)
## TODO
    # Set Learning Rate, Batch size, and Epoch number in .toml file (DONE)
    # Read in Training, Validation, and Test Data via DataLoader (DONE)
    # Select proper Optimizer and Loss function for task (DONE)
    # Set up data cache to store training results as they happen (DONE)
    # For each epoch:
        # Turn on training mode
        # Loop over all the training data in that epoch
            # Send data to correct device
            # perform forward propagation, then calculate loss function
            # zero out gradients, do backpropagation, then update weights with opt.step()
            # Add loss data to data cache
        # Once done with training data, evaulate on validation data
            # turn off gradient calcs
            # set model to eval mode
            # loop over validation data
            # do forward pass and calculate the loss function/ correct classification
            # Save to data cache
    # Do Validation step on test data once all epochs are done
    # Do plotting stuff
    # Save model to disk vis torch.save()