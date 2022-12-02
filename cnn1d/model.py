from torch.nn import Module
from torch import nn
import torch

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
        #self.Softmax = nn.Softmax(dim=1)


        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #     nn.Linear(512, features_number),
        # )
        #
        # self.regressor = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 1),
        # )

    # forward propagate input
    def forward(self, X):
        X = self.Layer1(X)
        # print(f'Layer1: {X.shape}')
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
        # print(f"X shape - before view: {X.shape}")
        # switch dimensions to fit the FC layer
        X = X.permute(0, 2, 1)
        # print(f"X shape - after view: {X.shape}")
        X = self.FC(X)
        # print(f"X shape - after FC: {X.shape}")
        X = torch.squeeze(X, dim=1)
        # print(f"X shape - after squeeze: {X.shape}")
        # print(f"X shape - after FC: {X.shape}")
        # X = self.Softmax(X)
        # X = X[None, :]
        # print(f"X shape - after None: {X.shape}")
        # X = self.decoder(X)
        # X = self.regressor(X)
        return X
