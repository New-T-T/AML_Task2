import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

# model definition
class MLP(Module):

    # define model elements
    def __init__(self, n_outputs):
        super(MLP, self).__init__()

        self.Layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=320, kernel_size=24, stride=1, padding="same", dilation=1),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.Layer2 = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.Layer7 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer8 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer9 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer10 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer11 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.Layer12 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=1, padding="same", dilation=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer13 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding="same", dilation=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Linear(64, n_outputs)

    # forward propagate input
    def forward(self, X):
        X = self.Layer1(X)
        X = self.Layer2(X)
        X = self.Layer3(X)
        X = self.Layer4(X)
        X = self.Layer5(X)
        X = self.Layer6(X)
        X = self.Layer7(X)
        X = self.Layer8(X)
        X = self.Layer9(X)
        X = self.Layer10(X)
        X = self.Layer11(X)
        X = self.Layer12(X)
        X = self.Layer13(X)
        X = self.GAP(X)
        X = X.permute(0, 2, 1)
        X = self.FC(X)
        X = torch.squeeze(X, dim=1)

        return X

# model definition
class MLP_reduced(Module):

    # define model elements
    def __init__(self):
        super(MLP_reduced, self).__init__()

        self.Layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=320, kernel_size=24, stride=1, padding="same", dilation=1),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.Layer2 = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.Layer7 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer8 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer9 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer10 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer11 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding="same", dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.Layer12 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=1, padding="same", dilation=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.Layer13 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding="same", dilation=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.GAP = nn.AdaptiveAvgPool1d(1)

    # forward propagate input
    def forward(self, X):
        X = self.Layer1(X)
        X = self.Layer2(X)
        X = self.Layer3(X)
        X = self.Layer4(X)
        X = self.Layer5(X)
        X = self.Layer6(X)
        X = self.Layer7(X)
        X = self.Layer8(X)
        X = self.Layer9(X)
        X = self.Layer10(X)
        X = self.Layer11(X)
        X = self.Layer12(X)
        X = self.Layer13(X)
        X = self.GAP(X)
        X = X.permute(0, 2, 1)

        return X

# Setting the seed for generating random numbers ensure that reproducible results
SEED = 42
torch.manual_seed(SEED)

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Google colab offers time limited use of GPU for free
print(device)

model = MLP(n_outputs=4)
model_reduced = MLP_reduced()

model.load_state_dict(torch.load('cnn1d/training_results/cnn1d_10_20_30/model.path', map_location=device))

#%%
state = model.state_dict()
# drop the last layer of the model
state_reduced = {k: v for k, v in state.items() if 'FC' not in k}
#%%
model_reduced.load_state_dict(state_reduced)
#%%
