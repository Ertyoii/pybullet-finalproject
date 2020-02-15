import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from data import *
from model import *

PATH = './NaiveNet.pth'


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
    train_data = RobotDataset(input, label)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(epochs, train_loader, optimizer, model, criterion)


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
    main()
    # test_load_model()
    # test_image_in_dataset()
