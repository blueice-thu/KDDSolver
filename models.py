import torch
import torch.utils.data
from TCN.tcn import TemporalConvNet


class CNNModel(torch.nn.Module):
    def __init__(self, n_features):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=n_features, out_channels=128, kernel_size=(1,)),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=1)
        )
        self.mlp1 = torch.nn.Linear(in_features=128, out_features=64)
        self.mlp2 = torch.nn.Linear(in_features=64, out_features=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, x.shape[1])
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.activation(x)
        x = x.view(-1)
        return x


class TCNModel(torch.nn.Module):
    def __init__(self, n_features):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs=n_features,
            num_channels=[2 ** i for i in range(7)],
            dropout=0.0
        )
        self.batch_normal = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(in_features=64, out_features=1)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias, 0)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.tcn(x)
        x = self.batch_normal(x)
        x = self.relu(x)
        x = x.view(-1, x.shape[1])
        x = self.linear(x)
        x = self.activation(x)
        x = x.view(-1)
        return x


class RNNModel(torch.nn.Module):
    def __init__(self, n_features):
        super(RNNModel, self).__init__()
        self.RNN = torch.nn.RNN(input_size=n_features, hidden_size=64, num_layers=2)
        self.mlp1 = torch.nn.Linear(in_features=64, out_features=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.RNN(x)[0]
        x = x.view(-1, x.shape[2])
        x = self.mlp1(x)
        x = self.activation(x)
        x = x.view(-1)
        return x


class LSTMModel(torch.nn.Module):
    def __init__(self, n_features):
        super(LSTMModel, self).__init__()
        self.LSTM = torch.nn.LSTM(input_size=n_features, hidden_size=32, num_layers=2)
        self.mlp1 = torch.nn.Linear(in_features=32, out_features=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.LSTM(x)[0]
        x = x.view(-1, x.shape[2])
        x = self.mlp1(x)
        x = self.activation(x)
        x = x.view(-1)
        return x
