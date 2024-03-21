import torch
from torch import nn
from torch.utils.data import DataLoader
from AnodePlaneDataset import AnodePlaneDataset
import matplotlib.pyplot as plt
import numpy as np

import sparseconvnet as scn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

from safetensors.torch import save_model, load_model

import sys, itertools
import argparse

from dataclasses import dataclass, field

sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck

class SimpleNN(nn.Module):
# Bare-bones Neural Network Architecture. Need to futzs around with this
    def __init__(self, PixelCountX, PixelCountY, output="class"):
        self.net_type = output.lower()
        super().__init__()
        if(self.net_type =="class"):
            self.EscapeStack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features= PixelCountX*PixelCountY, out_features=512),
                nn.ReLU(),
                nn.Linear(512,1),
                nn.Sigmoid()
            )
        elif(self.net_type == "energy"):
            self.EscapeStack = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features= PixelCountX*PixelCountY, out_features=512),
                nn.ReLU(),
                nn.Linear(512,1),
            )
        else:
            raise Exception("Invalid output class")

    def forward(self,x):
        return self.EscapeStack(x)

class SimpleCNN(nn.Module):
# Bare-bones Neural Network Architecture for classification. Need to futzs around with this
    def __init__(self, PixelCountX, PixelCountY, output="class"):
        self.net_type = output.lower()
        super().__init__()
        if(self.net_type=="class"):
            self.EscapeStack = nn.Sequential(
    # feature extraction
    # https://madebyollin.github.io/convnet-calculator/ (Calculating intermediate layer of convolution)
                nn.Conv2d(1,32,3),
                nn.ReLU(),
                nn.MaxPool2d(3,3),
                nn.Conv2d(32, 16, 3),
                nn.ReLU(),
                nn.MaxPool2d(3,3),
    # Classification
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.ReLU(),
                nn.Sigmoid()
            )
        elif(self.net_type=="energy"):
            self.EscapeStack = nn.Sequential(
    # feature extraction
    # https://madebyollin.github.io/convnet-calculator/ (Calculating intermediate layer of convolution)
                nn.Conv2d(1,32,3),
                nn.ReLU(),
                nn.MaxPool2d(3,3),
                nn.Conv2d(32, 16, 3),
                nn.ReLU(),
                nn.MaxPool2d(3,3),
    # Classification
                nn.Flatten(),
                nn.Linear(64, 1)
            )
        else:
            raise Exception("Invalid output class")

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
        plt.close()
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
                raise Exception("which in History.plot() must be either 'loss' or 'acc'")
        plt.close()

@dataclass
class HyperParameters:
# Effectively a struct of hyperparameters
    learning_rate: float
    batch_size: int
    epoch_num: int

class Trainer:
# Used to encapsulate the training loop of the Neural Net
    def __init__(self, parameters,optimizer, loss_func, max_f=None, output_type="class", model_type="simple"):
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
                                    int(parameters["TrainData"]["NNBatchSize"]["value"]),
                                    int(parameters["TrainData"]["EpochNum"]["value"])
        )
# Create Net and send to proper device
        PixelCountX = int(parameters["GenData"]["PixelCountX"]["value"])
        PixelCountY = int(parameters["GenData"]["PixelCountY"]["value"])
        match model_type.lower().strip():
            case "simple":
                self.model = SimpleNN(PixelCountX,PixelCountY,output_type)
            case "cnn":
                self.model = SimpleCNN(PixelCountX,PixelCountY,output_type)
            case _:
                raise Exception("invalid model type")
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
    
    def predict_all(self, which_data="test"):
        data = None
        match which_data:
            case "test":
                data = self.test_data
            case "val":
                data = self.val_data
            case _:
                raise Exception("Invalid Dataloader")
        predictions = []
        truth_level = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data):
                inpt, lbls = batch
                truth_labels = None
                match self.model.net_type:
                    case "class":
                        # Label denoting escape or all in event
                        truth_labels = lbls[:,:,2]
                    case "energy":
                        # 0 is energy of initial gamma ray
                        truth_labels = lbls[:,:,0]
                    case _:
                        raise Exception("Undefined loss target")
                voutputs = self.model(inpt).round()
                for j in range(truth_labels.shape[0]):
                    truth_level.append(truth_labels[j][0].item())
                    predictions.append(voutputs[j][0])
        return (truth_level, predictions)

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
                match self.model.net_type:
                    case "class":
                        # Label denoting escape or all in event
                        truth_labels = labels[:,:,2]
                    case "energy":
                        # 0 is energy of initial gamma ray
                        truth_labels = labels[:,:,0]
                    case _:
                        raise Exception("Undefined loss target")
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
                    if(self.model.net_type=="class"):
# Round sigmoid output to nearest number, then compare to truth labels. Sum counts and add to correct
                        correct += ( voutputs.round() == vtruth_labels).type(torch.float).sum().item()
# Update training history with data from current epoch
            if(self.model.net_type=="class"):
                self.training_history.add_loss(train_running_loss, val_run_loss)
                self.training_history.add_accuracy(correct/total)
            else:
                self.training_history.add_loss(train_running_loss, val_run_loss)
            if logging:
                    if(self.model.net_type=="class"):
                        print("Epoch:", epoch,"Training: ", train_running_loss,"Validation: ", val_run_loss, "correct:", correct, "Percent:", correct/total)
                    else:
                        print("Epoch:", epoch,"Training: ", train_running_loss,"Validation: ", val_run_loss)
    def save_model(self, filename:str):
        save_model(self.model,filename)
    def load_model(self, filename: str):
        load_model(self.model, filename)

class Plotter:
    def __init__(self):
        self.cmap = plt.get_cmap('GnBu')
    def plot_confusion_mat_scipy(self,predictions,truth, title):
        cm = confusion_matrix(truth,predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(title)
        plt.savefig(title+".png")
        plt.close()
    def plot_confusion_mat(self, predictions, truth, class_names = None,
        title="Confusion matrix", normalize=False, save=True):
        cm = confusion_matrix(truth,predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=self.cmap, vmin=0, vmax=1)
        plt.title(title)
        plt.colorbar()

        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, fontsize=15)
            plt.yticks(tick_marks, class_names,fontsize=15)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Add percentage/count to center of each cell in confusion matrix, choosing color for better visibility
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                    horizontalalignment="center", fontsize=15,
                    color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)

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
    print(paras)
    trainer = Trainer(paras, torch.optim.Adam, nn.BCELoss, int(paras["TrainData"]["MaxFiles"]["value"]),
                      output_type=paras["TrainData"]["Target"]["value"], model_type=paras["TrainData"]["NetworkType"]["value"])
    temp_plotter = Plotter()
    truth, pred = trainer.predict_all()
#    temp_plotter.plot_confusion_mat(pred, truth, class_names = ["AllIn","Escape"])
    temp_plotter.plot_confusion_mat_scipy(pred, truth, "Prior to Training")
    trainer.fit()
    trainer.training_history.dump()
    trainer.training_history.plot("loss")
    if(paras["TrainData"]["Target"]["value"].lower().strip()=="class"):
        trainer.training_history.plot("acc")
    truth, pred = trainer.predict_all()
#    temp_plotter.plot_confusion_mat(pred, truth, class_names = ["AllIn","Escape"])
    temp_plotter.plot_confusion_mat_scipy(pred, truth, "After Training")
    trainer.save_model(paras["TrainData"]["ModelFile"]["value"])