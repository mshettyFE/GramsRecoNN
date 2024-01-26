import torch
from torch import nn
from torch.utils.data import DataLoader
from AnodePlaneDataset import AnodePlaneDataset
import matplotlib.pyplot as plt
import numpy as np

from safetensors.torch import save_model, load_model

import sys
import argparse

from dataclasses import dataclass, field

sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck

class SimpleNN(nn.Module):
# Bare-bones Neural Network Architecture. Need to futzs around with this
    def __init__(self, PixelCountX, PixelCountY):
        super().__init__()
        self.EscapeStack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= PixelCountX*PixelCountY, out_features=512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.EscapeStack(x)

@dataclass
class History:
# Stores data on how training is going on a epoch by epoch basis
    training_loss_hist: list[float] = field(default_factory=list)
    validation_loss_hist: list[float] = field(default_factory=list)
    loss_epoch_count: list[int] = field(default_factory=list)
    validation_accuracy_history: list[float] = field(default_factory=list)
    acc_epoch_count: list[int] = field(default_factory=list)
    def add_loss(self,training_loss: float, validation_loss: float, epoch_num:int =None):
# training_loss and validation_loss are just doubles
# Can specify what epoch number these losses correspond to
# If you don't, the epoch number is just the total number of losses recorded (0-based index)
        if epoch_num==None:
            self.loss_epoch_count.append(len(self.training_loss_hist))
        else:
            self.loss_epoch_count.append(epoch_num)
        self.training_loss_hist.append(training_loss)
        self.validation_loss_hist.append(validation_loss)

    def add_accuracy(self, validation_percentage: float, epoch_num:int =None):
# Same as loss, but input is a percentage of how many labels were accurately classified. Only for validation set
        if epoch_num==None:
            self.acc_epoch_count.append(len(self.validation_accuracy_history))
        else:
            self.acc_epoch_count.append(epoch_num)
        self.validation_accuracy_history.append(validation_percentage)

    def print(self):
# Helper function to print out logs
        print("Loss")
        print(self.loss_epoch_count)
        print(self.training_loss_hist)
        print(self.validation_loss_hist)
        print("Accuracy")
        print(self.acc_epoch_count)
        print(self.validation_accuracy_history)

    def dump(self, filename="Dump.csv"):
# Dump data to ill-formatted csv file (Useful for debugging. Don't actually use this to save training runs for the long term)
        with open(filename,'w') as f:
            f.write("Loss\n")
            for i in range(len(self.loss_epoch_count)):
                f.write(",".join( [str(x) for x in   [self.loss_epoch_count[i],self.training_loss_hist[i], self.validation_loss_hist[i]] ]  )+"\n")
            f.write("Acc\n")
            for i in range(len(self.acc_epoch_count)):
                f.write(",".join([str(x) for x in [self.acc_epoch_count[i], self.validation_accuracy_history[i]]])+"\n")

    def plot(self, which="loss"):
# saves .pngs of either losses or accuracy 
        match which:
            case "loss":
                x = np.array(self.loss_epoch_count)
                train = np.array(self.training_loss_hist)
                val = np.array(self.validation_loss_hist)
                plt
                plt.scatter(x, train)
                plt.scatter(x, val, color="red")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title('Loss over Epochs')
                plt.savefig('LossRecord.png')
            case "acc":
                x = np.array(self.acc_epoch_count)
                val = np.array(self.validation_accuracy_history)
                plt.scatter(x, val, color="red")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy (%)")
                plt.title('Accuracy over Epochs')
                plt.savefig('AccuracyRecord.png')
            case _:
                raise Exception("which must be either 'loss' or 'acc'")
        plt.close()

@dataclass
class HyperParameters:
# Effectively a struct of hyperparameters
    learning_rate: float
    batch_size: int
    epoch_num: int

class Trainer:
# Used to encapsulate the training loop of the Neural Net
    def __init__(self, parameters,optimizer, loss_func, max_f=None):
# If you can run on GPU, do so.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
# Cast Hyperparameters to the appropriate values
        self.args = HyperParameters(
            float(parameters["TrainData"]["LearningRate"]["value"]),
                                    int(parameters["TrainData"]["BatchSize"]["value"]),
                                    int(parameters["TrainData"]["EpochNum"]["value"])
        )
# Create Net and send to proper device
        PixelCountX = int(parameters["GenData"]["PixelCountX"]["value"])
        PixelCountY = int(parameters["GenData"]["PixelCountY"]["value"])
        self.model = SimpleNN(PixelCountX,PixelCountY)
        self.model.to(self.device)
# Make PyTorch stop complaining about incompatibility between floats and doubles
        self.model.double()
# Construct optimizer, loss function, load in data, and create History buffer
        self.opt = optimizer(self.model.parameters(),  lr=self.args.learning_rate)
        self.loss = loss_func()
        self.train_data = DataLoader(AnodePlaneDataset(parameters["TrainData"]["InputTrainingFolderPath"]["value"], max_files=max_f),
                                 batch_size=self.args.batch_size, shuffle=True)
        self.val_data = DataLoader(AnodePlaneDataset(parameters["TrainData"]["InputValidationFolderPath"]["value"], max_files=max_f),
                              batch_size=self.args.batch_size, shuffle=False)
        self.test_data = DataLoader(AnodePlaneDataset(parameters["TrainData"]["InputTestFolderPath"]["value"], max_files=max_f),
                               batch_size=self.args.batch_size, shuffle=False)
        self.training_history = History()

    def fit(self, logging=True):
# Fit the model to the data
        for epoch in range(self.args.epoch_num):
# Turn training mode on (needed since eval mode is turned on at end of last cycle)
            self.model.train(True)
            train_running_loss  = 0.0
            prev_loss = 0.0
# Run through all the batches
            for i,batch in enumerate(self.train_data):
                inputs, labels = batch
# Grab truth level labels (remember, output is Energy, Angle, Classification)
                truth_labels = labels[:,:,2]
                inputs.to(self.device)
                truth_labels.to(self.device)
# Do the backpropagation
                outputs = self.model(inputs)
                loss = self.loss(outputs, truth_labels)
                loss.backward()
                self.opt.step()
# Clean out the gradients for the next cycle
                self.opt.zero_grad()
# Update loss of current epoch
                train_running_loss += loss.item()
                if((i%1000)==0 and logging and i!=0):
                    print(epoch,i,train_running_loss-prev_loss)
                    prev_loss = train_running_loss
            val_run_loss = 0.0
# Do evaluate phase (ie. run on validation data)
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, vdata in enumerate(self.val_data):
                    vinputs, vlabels = vdata
                    vtruth_labels = vlabels[:,:,2]
                    voutputs = self.model(vinputs)
                    total += vtruth_labels.shape[0]
                    val_run_loss += self.loss(voutputs, vtruth_labels).item()
# Round sigmoid output to nearest number, then compare to truth labels. Sum counts and add to correct
                    correct += ( voutputs.round() == vtruth_labels).type(torch.float).sum().item()
# Update training history with data from current epoch
            self.training_history.add_loss(train_running_loss, val_run_loss)
            self.training_history.add_accuracy(correct/total)
            if logging:
                print("Epoch:", epoch,"Training: ", train_running_loss,"Validation: ", val_run_loss, "correct:", correct, "Percent:", correct/total)
    def save_model(self, filename:str):
        save_model(self.model,filename)
    def load_model(self, filename: str):
        load_model(self.model, filename)


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
    trainer = Trainer(paras, torch.optim.Adam, nn.BCELoss,1)
    trainer.fit()
    trainer.training_history.dump()
    trainer.training_history.plot("loss")
    trainer.training_history.plot("acc")
    trainer.save_model(paras["GenData"]["ModelFile"]["value"])