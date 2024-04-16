# Pytorch imports
import torch
from torch import optim
from torch import nn
from torch import no_grad, cuda, backends
from torch.utils.data import DataLoader, WeightedRandomSampler

# Standard scientific python packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
# Used to compare truth/prediction distributions
from scipy.stats import ks_2samp

# Custom Dataset to load .safetensor files from a folder in a way that DataLoader can understand
from AnodePlaneDataset import AnodePlaneDataset
from safetensors.torch import save_model, load_model

# SparseCNN implementation
import sparseconvnet as scn

# Used for decorators of Hyperparameter and other classes
from dataclasses import dataclass, field

# STD library functions
import sys, itertools
import argparse

# Include path immediately above to import TomlSanityCheck
sys.path.append('..')
from  TomlSanityCheck import TomlSanityCheck

class SimpleNN(nn.Module):
# Bare-bones Neural Network Architecture. Don't actually use this for the real network. Just for testing pipeline
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
                nn.Dropout(),
                nn.Linear(64, 1),
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
    # Regression
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

    def plot(self, which="loss",prefix_title=""):
# saves .pngs of either losses or accuracy
        plt.close()
        match which:
            case "loss":
                title = prefix_title+"_LossRecord.png"
                x = np.array(self.loss_epoch_count)
                train = np.array(self.training_loss_hist)
                val = np.array(self.validation_loss_hist)
                plt
                plt.scatter(x, train)
                plt.scatter(x, val, color="red")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title('Loss over Epochs')
                plt.savefig(title)
            case "acc":
                title = prefix_title+"AccuracyRecord.png"
                x = np.array(self.acc_epoch_count)
                val = np.array(self.validation_accuracy_history)
                plt.scatter(x, val, color="red")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy (%)")
                plt.title('Accuracy over Epochs')
                plt.savefig(title)
            case _:
                raise Exception("which in History.plot() must be either 'loss' or 'acc'")
        plt.close()

@dataclass
class HyperParameters:
# Effectively a struct of hyperparameters
    learning_rate: float
    batch_size: int
    epoch_num: int
    L2_regularization: float

@dataclass
# Store the average energies of the Training, Validation, and Test for all in and escape
class Average_Energies:
    train_all_in: float
    train_escape: float
    validate_all_in: float
    validate_escape: float
    test_all_in: float
    test_escape: float

class Trainer:
# Used to encapsulate the training loop of the Neural Net
    def __init__(self, parameters,optimizer, loss_func, max_f=None, output_type="class", model_type="simple"):
# If you can run on GPU, do so.
        self.device = (
            "cuda"
            if cuda.is_available()
            else "mps"
            if backends.mps.is_available()
            else "cpu"
        )
        print(self.device)
# Cast Hyperparameters to the appropriate values
        self.args = HyperParameters(
            float(parameters["TrainData"]["LearningRate"]["value"]),
            int(parameters["TrainData"]["NNBatchSize"]["value"]),
            int(parameters["TrainData"]["EpochNum"]["value"]),
            float(parameters["TrainData"]["L2Reg"]["value"])
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
# Make PyTorch stop complaining about incompatibility between floats and doubles
        self.model.double()
        self.model.to(self.device)
# Construct optimizer, loss function, load in data, and create History buffer
        self.opt = optimizer(self.model.parameters(),  lr=self.args.learning_rate, weight_decay=self.args.L2_regularization)
        self.loss = loss_func()
# This part just reads in the Dataset(s)
# Also, get average energy for each subset. Can be used as a baseline to help debug regression predictions
        train_dataset = AnodePlaneDataset(parameters["TrainData"]["InputTrainingFolderPath"]["value"], max_files=max_f)
        train_avg_energies = (train_dataset.avg_energy_all_in, train_dataset.avg_energy_escape)
        val_dataset = AnodePlaneDataset(parameters["TrainData"]["InputValidationFolderPath"]["value"], max_files=max_f)
        val_avg_energies = (val_dataset.avg_energy_all_in, val_dataset.avg_energy_escape)
        test_dataset = AnodePlaneDataset(parameters["TrainData"]["InputTestFolderPath"]["value"], max_files=max_f)
        test_avg_energies = (test_dataset.avg_energy_all_in, test_dataset.avg_energy_escape)
        self.avg_energies = Average_Energies(
            train_avg_energies[0], train_avg_energies[1],
            val_avg_energies[0], val_avg_energies[1],
            test_avg_energies[0], test_avg_energies[1],
        )
# Actually saving Datasets to class
        self.train_data = DataLoader(train_dataset, sampler=WeightedRandomSampler(weights=train_dataset.emit_weight_map_data()[0],num_samples=len(train_dataset),replacement=True),batch_size=self.args.batch_size)
        self.val_data = DataLoader(val_dataset,batch_size=self.args.batch_size)
        self.test_data = DataLoader(test_dataset,batch_size=self.args.batch_size)
        self.training_history = History()
    
    def predict_all(self, which_data="test"):
# Given the current state of the model, do prediction on all of the data in a given set (test or validation)
        data = None
        match which_data:
            case "test":
                data = self.test_data
            case "val":
                data = self.val_data
            case _:
                raise Exception("Invalid Dataloader")
# Place to store output
        predictions = []
        truth_level = []
        self.model.eval()
        with no_grad():
            for i, batch in enumerate(data):
                inpt, lbls = batch
                truth_labels = None
                match self.model.net_type:
                    case "class":
                        # Label denoting escape or all in event. index 2 is where classification label is
                        truth_labels = lbls[:,:,2]
                        inpt = inpt.to(self.device)
                        voutputs = self.model(inpt).round()
                        for j in range(truth_labels.shape[0]):
                            truth_level.append(truth_labels[j][0].item())
                            predictions.append(voutputs[j][0])
                    case "energy":
                        # 0 is energy of initial gamma ray
                        truth_labels = lbls[:,:,0]
                        inpt = inpt.to(self.device)
                        voutputs = self.model(inpt)
                        for j in range(truth_labels.shape[0]):
                            truth_level.append(truth_labels[j][0].item())
                            predictions.append(voutputs[j][0])
                    case _:
                        raise Exception("Undefined loss target")
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
                inputs =inputs.to(self.device)
                truth_labels =  truth_labels.to(self.device)
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
            with no_grad():
                for i, vdata in enumerate(self.val_data):
                    vinputs, vlabels = vdata
                    vtruth_labels = vlabels[:,:,2]
                    vinputs = vinputs.to(self.device)  
                    vtruth_labels = vtruth_labels.to(self.device)
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
                        print("Epoch:", epoch,"Training: ", train_running_loss,"Validation: ", val_run_loss, "correct:", correct, "total: ",  total, "Percent:", correct/total)
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
        predictions = [x.cpu() for x in predictions] # Transfer tensor from gpu to cpu
        cm = confusion_matrix(truth,predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.clf()
        disp.plot()
        plt.title(title)
        plt.savefig(title+".png")
        plt.close()
    def plot_regression_scatter(self,predictions,truth, title):
        predictions = np.array([x.cpu() for x in predictions]) # Transfer tensor from gpu to cpu
        MSE_Error = sum(np.abs(predictions-truth))
        plt.clf()
        plt.scatter(truth, predictions)
        plt.plot(truth, truth, color = 'red')
        plt.xlabel("Truth Gamma Energy")
        plt.ylabel("Prediction Gamma Energy")
        plt.title(title+" MSE_Error="+str(MSE_Error))
        plt.savefig(title+"_Scatter.png")
        plt.close()
    def plot_regression_histograms(self, predictions, truth,title):
        predictions = np.array([x.cpu() for x in predictions]) # Transfer tensor from gpu to cpu
        MSE_Error = sum(np.abs(predictions-truth))
        plt.clf()
        plt.hist(truth,bins=100)
        plt.xlabel("Truth Gamma Energy")
        plt.ylabel("Count")
        plt.title(title+" MSE_Error="+str(MSE_Error))
        plt.savefig(title+"_Hist_Truth.png")
        plt.close()
        plt.clf()
        plt.hist(predictions, bins=100)
        plt.xlabel("Truth Gamma Energy")
        plt.ylabel("Count")
        KS_test_stat, KS_test_p_val = ks_2samp(truth, predictions)
        plt.title(title+" Stat:"+str(KS_test_stat)+" Pval"+str(KS_test_p_val))
        plt.savefig(title+"_Hist_Pred.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TrainNet')
    parser.add_argument('TOML',help="Path to .toml config file")
    args = parser.parse_args()
# Sanity check config files
    sanity_checker = TomlSanityCheck(args.TOML)
    try:
        sanity_checker.validate()
    except Exception as e:
        print(e)
        sys.exit()
    paras=  sanity_checker.return_config()
    print(paras)
# Spin up Trainer depending on the classification task at hand
    target  = paras["TrainData"]["Target"]["value"].lower().strip()
    titles = target+"_Prior_Training"
    if(target=="class"):
        trainer = Trainer(paras, optim.Adam, nn.BCELoss, int(paras["TrainData"]["MaxFiles"]["value"]),
                        output_type=paras["TrainData"]["Target"]["value"], model_type=paras["TrainData"]["NetworkType"]["value"])
        temp_plotter = Plotter()
# Also, make prediction without any training as baseline
        truth, pred = trainer.predict_all()
        temp_plotter.plot_confusion_mat_scipy(pred, truth, titles)
    elif(target=="energy"):
        trainer = Trainer(paras, optim.Adam, nn.MSELoss, int(paras["TrainData"]["MaxFiles"]["value"]),
                        output_type=paras["TrainData"]["Target"]["value"], model_type=paras["TrainData"]["NetworkType"]["value"])
        temp_plotter = Plotter()
        truth, pred = trainer.predict_all()
        temp_plotter.plot_regression_scatter(pred,truth, titles)
        temp_plotter.plot_regression_histograms(pred,truth,titles)
    else:
        raise Exception("Invalid initial class")
# Learn damn it!
    trainer.fit()
# Write out diagnostics of training arc
    trainer.training_history.dump()
    trainer.training_history.plot("loss", prefix_title=titles)
    truth, pred = trainer.predict_all()
    titles = target+"_After_Training"
    if(target=="class"):
        trainer.training_history.plot("acc", prefix_title=titles)
        temp_plotter.plot_confusion_mat_scipy(pred, truth, titles)
    elif(target=="energy"):
        temp_plotter.plot_regression_scatter(pred,truth, titles)
        temp_plotter.plot_regression_histograms(pred,truth,titles)
    else:
        raise Exception("Invalid prediction task")
# Save model for later...
    trainer.save_model(paras["TrainData"]["ModelFile"]["value"])
