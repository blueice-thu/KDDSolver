import pandas as pd
import numpy as np
import torch
import torch.utils.data
from datetime import datetime
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import CNNModel, RNNModel, TCNModel, LSTMModel
from Log import Log

plt.ion()
# device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = Log()


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
        log.write("============ epoch {} / {} ============".format(epoch + 1, n_epochs))
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
        log.write("time cost: " + str(end_time - start_time))
        log.write("train accuracy = %.2f%%" % (score / total * 100))
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
    log.write("{} / {}".format(correct, total), end=", ")
    log.write("test accuracy = %.2f%%" % (correct / total * 100))


if __name__ == '__main__':
    n_features = 28
    n_epochs = 5
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
        ['dataset/KDDTrain+/x.csv',         ],
        ['dataset/KDDTrain+/y_U2R.csv',   ]
    )
    test_dataset = KDDDataset(
        ["dataset/KDDTest+/x.csv"       ],
        ["dataset/KDDTest+/y_U2R.csv"   ]
    )
    export_dataset = KDDDataset('dataset/NormalData/x.csv', 'dataset/NormalData/y.csv')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    export_data_loader = torch.utils.data.DataLoader(export_dataset, batch_size=batch_size, shuffle=True)

    train(model, train_data_loader, test_data_loader, n_epochs)
    test(model, test_data_loader)

    time_str = str(datetime.now()).split(".")[0].replace("-", "_").replace(":", "_").replace(" ", "_")
    plt.savefig('./logs/loss_{}_{}.png'.format(model_name, time_str))
    torch.save(obj=model.state_dict(), f="logs/{}_{}.pth".format(model_name, time_str))

    # model.load_state_dict(torch.load("logs/LSTM.pth"))

    # test(model, export_data_loader)
