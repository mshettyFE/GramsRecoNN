import torch
from torch import nn
from torch.utils.data import DataLoader
from AnodePlaneDataset import AnodePlaneDataset
import matplotlib.pyplot as plt

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
    print(device)
    model = SimpleNN(10,10).to(device)
    print(model)
## TODO
    # Set Learning Rate, Batch size, and Epoch number in .toml file
    # Read in Training, Validation, and Test Data via DataLoader