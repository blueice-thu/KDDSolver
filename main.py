import pandas as pd
import numpy as np
import torch
import torch.utils.data
from datetime import datetime
from torch.autograd import Variable
import matplotlib.pyplot as plt
from TCN.tcn import TemporalConvNet

plt.ion()
# device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KDDDataset(torch.utils.data.Dataset):
    def __init__(self, x_csv, y_csv):
        if type(x_csv) == str:
            original_x = pd.read_csv(x_csv)
        elif type(x_csv) == list:
            original_x = pd.concat([pd.read_csv(csv) for csv in x_csv], axis=0)
        else:
            raise NotImplementedError()

        if type(y_csv) == str:
            original_y = pd.read_csv(y_csv)
        elif type(y_csv) == list:
            original_y = pd.concat([pd.read_csv(csv) for csv in y_csv], axis=0)
        else:
            raise NotImplementedError()

        self.x = np.expand_dims(original_x.to_numpy(), axis=2).astype(np.float64)
        self.y = np.expand_dims(original_y.to_numpy(), axis=2).astype(np.float64)
        self.n_data = self.x.shape[0]

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        return self.x[item], self.y[item]


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


def train(model, train_data_loader, test_data_loader, n_epochs):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1, last_epoch=-1)

    indexes = []
    losses = []
    index = 0
    for epoch in range(n_epochs):
        model.train()
        start_time = datetime.now()
        print("============ epoch {} / {} ============".format(epoch + 1, n_epochs))
        running_loss = 0.0
        total = 0.0
        score = 0.0
        for i, (x, y) in enumerate(train_data_loader):
            x, y = Variable(x).to(device), Variable(y).to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += y_pred.shape[0]
            score += ((y_pred.round() == y.view(-1)).sum().item())

            if i % 20 == 19:
                indexes.append(index)
                losses.append(running_loss)
                index += 1
                running_loss = 0.0
                plt.cla()
                plt.plot(indexes, losses)
                plt.pause(0.033)
                plt.show()
        scheduler.step()

        end_time = datetime.now()
        print("time cost: ", end_time - start_time)
        print("train accuracy = %.2f%%" % (score / total * 100))
        test(model, test_data_loader)


def test(model, test_data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for (x, y) in test_data_loader:
            x, y = Variable(x).to(device), Variable(y).to(device)
            y_pred = model(x)
            total += y.shape[0]
            correct += ((y_pred.round() == y.view(-1)).sum().item())
    print("{} / {}".format(correct, total), end=", ")
    print("test accuracy = %.2f%%" % (correct / total * 100))


if __name__ == '__main__':
    n_features = 28
    n_epochs = 10
    batch_size = 128
    model_name = 'LSTM'

    if model_name == '1DCNN':
        model = CNNModel(n_features)
    elif model_name == 'TCN':
        model = TCNModel(n_features)
    elif model_name == 'RNN':
        model = RNNModel(n_features)
    elif model_name == 'LSTM':
        model = LSTMModel(n_features)
    else:
        raise NotImplementedError('No such model: ', model_name)

    model = model.double().to(device)

    train_dataset = KDDDataset(
        ['dataset/KDDTrain+_binary/x.csv', "dataset/KDDTest+_binary/x.csv", 'dataset/ExportData/x.csv'],
        ['dataset/KDDTrain+_binary/y.csv', "dataset/KDDTest+_binary/y.csv", 'dataset/ExportData/y.csv']
    )
    test_dataset = KDDDataset(
        ["dataset/KDDTest+_binary/x.csv"],
        ["dataset/KDDTest+_binary/y.csv"]
    )
    export_dataset = KDDDataset('dataset/ExportData/x.csv', 'dataset/ExportData/y.csv')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    export_data_loader = torch.utils.data.DataLoader(export_dataset, batch_size=batch_size, shuffle=True)

    train(model, train_data_loader, test_data_loader, n_epochs)
    test(model, test_data_loader)

    time_str = str(datetime.now()).split(".")[0].replace("-", "_").replace(":", "_").replace(" ", "_")
    plt.savefig('./train_info/loss_{}_{}.png'.format(model_name, time_str))
    torch.save(obj=model.state_dict(), f="train_info/{}_{}.pth".format(model_name, time_str))

    # model.load_state_dict(torch.load("train_info/LSTM.pth"))

    test(model, export_data_loader)
