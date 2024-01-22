import torch
from torch import nn
from torch.utils.data import DataLoader
from AnodePlaneDataset import AnodePlaneDataset
import matplotlib.pyplot as plt

import sys
import argparse

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

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser = argparse.ArgumentParser(prog='TrainNet')
    parser.add_argument('TOML',help="Path to .toml config file")
    args = parser.parse_args()
    sanity_checker = TomlSanityCheck(args.TOML)
    try:
        sanity_checker.validate()
    except Exception as e:
        print(e)
        sys.exit()
    parameters=  sanity_checker.return_config()
    learning_rate = float(parameters["TrainData"]["LearningRate"]["value"])
    PixelCountX = int(parameters["GenData"]["PixelCountX"]["value"])
    PixelCountY = int(parameters["GenData"]["PixelCountY"]["value"])
    model = SimpleNN(PixelCountX,PixelCountY).to(device)
    opt = torch.optim.Adam(model.parameters(),  lr=learning_rate)
    loss = nn.NLLLoss()
    train_data = AnodePlaneDataset(parameters["TrainData"]["InputTrainingFolderPath"]["value"])
    val_data = AnodePlaneDataset(parameters["TrainData"]["InputValidationFolderPath"]["value"])
    test_data = AnodePlaneDataset(parameters["TrainData"]["InputTestFolderPath"]["value"])
## TODO
    # Set Learning Rate, Batch size, and Epoch number in .toml file (DONE)
    # Read in Training, Validation, and Test Data via DataLoader (DONE)
    # Select proper Optimizer and Loss function for task (DONE)
    # Set up data cache to store training results as they happen
    # For each epoch:
        # Turn on training mode
        # Loop over all the training data in that epoch
            # Send data to correct device
            # perform forward propagation, then calculate loss function
            # zero out gradients, do backpropagation, then update weights with opt.step()
            # Add loss data to data cache
        # Once done with training data, evaulate on validation data
            # turn of gradient calcs
            # set model to eval mode
            # loop over validation data
            # do forward pass and calculate the loss function/ correct classification
            # Save to data cache
    # Do Validation step on test data once all epochs are done
    # Do plotting stuff
    # Save model to disk vis torch.save()