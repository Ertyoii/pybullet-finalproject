import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


PATH = './NaiveNet.pth'


class RobotDataset(Dataset):
    def __init__(self, input, target):
        self.input = torch.from_numpy(input).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx):
        x = self.input[idx]
        label = self.target[idx]
        sample = {'X': x, 'l': label}
        return sample

    def __len__(self):
        return len(self.input)


class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 64 * 64, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test():
    dtype = torch.float32
    x = torch.zeros((64, 1, 256, 256), dtype=dtype)
    y = torch.ones((64, 3), dtype=dtype)

    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for _ in range(5):
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        print(outputs)

    torch.save(model.state_dict(), PATH)


def train(epochs, train_loader, optimizer, model, criterion):
    iter = 0
    for epoch in range(epochs):
        for _, batch in enumerate(train_loader):
            iter = iter + 1

            optimizer.zero_grad()
            inputs, labels = Variable(batch['X']), Variable(batch['l'])
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if iter % 2 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), PATH)


def main():
    epochs = 100

    input, label = load_data()
    train_data = RobotDataset(input[:100, :, :, :], label[:100, :])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(epochs, train_loader, optimizer, model, criterion)


def load_data():
    f = h5py.File("data.h5", 'r')
    input = np.array(f["input"]).reshape(10000, 1, 256, 256)
    label = np.array(f["label"])
    f.close()

    return input, label


def test_load_model():
    net = NaiveNet()
    net.load_state_dict(torch.load(PATH))

    input, label = load_data()

    first_input = torch.from_numpy(input[0].reshape(1, 1, 256, 256)).float()
    print(net(first_input))
    print(label[0])


def test_image_in_dataset():
    input, _ = load_data()
    img = input[2].reshape(256, 256)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # test()
    # main()
    # test_load_model()
    test_image_in_dataset()
