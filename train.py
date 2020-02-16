import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import *
from model import *


def train(epochs, train_loader, use_gpu, optimizer, model, criterion, save_dir):
    for epoch in range(epochs):
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['l'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['l'])

            outputs = model(inputs)
            x_mul = torch.ones(outputs.shape) * 10
            y_mul = torch.ones(labels.shape) * 10
            if use_gpu:
                x_mul = x_mul.cuda()
                y_mul = y_mul.cuda()

            outputs = torch.mul(x_mul, outputs)
            labels = torch.mul(y_mul, labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # print("loss: {}".format(loss.item()))

        if epoch % 10 == 0:
            print("epoch{}, loss: {}".format(epoch, loss.item()))

        if epoch % 100 == 0:
            filename = 'NaiveNet' + str(epoch) + '.pth'
            save_path = os.path.join(save_dir, filename)
            torch.save(model.state_dict(), save_path)


def main(data_path, save_dir):
    epochs = 1000
    batch_size = 64
    learning_rate = 4e-4

    f = h5py.File(data_path, 'r')
    images = np.array(f["input"]).reshape(10000, 1, 256, 256)
    label = np.array(f["label"])
    f.close()

    data_size = 256
    train_data = RobotDataset(images[:data_size, :, :, :], label[:data_size, :])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model.cuda()

    train(epochs, train_loader, use_gpu, optimizer, model, criterion, save_dir)


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: {} data_path SAVE_DIR".format(sys.argv[0]))
        sys.exit(1)

    data_path = sys.argv[1]
    save_dir = sys.argv[2]

    main(data_path, save_dir)
