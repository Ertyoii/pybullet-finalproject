import os
import argparse
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import *
from model import *

parser = argparse.ArgumentParser(
    description='Hyperparameters and locations'
)
parser.add_argument('--data_path', type=str, required=True, help='path of h5 data file')
parser.add_argument('--save_dir', type=str, required=True, help='path to save checkpoints')
parser.add_argument('--learning_rate', '--lr', type=float, required=True, help='learning rate')
parser.add_argument('--epochs', '-ep', type=int, default=1000, help='epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=64, help='batch size')
parser.add_argument('--training_size', type=int, default=256, help='size of training data')


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

            if use_gpu:
                outputs = outputs.cuda()
                labels = labels.cuda()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            print("loss: {}".format(loss.item()))

        if epoch % 10 == 0:
            print("epoch{}, loss: {}".format(epoch, loss.item()))

        if epoch % 100 == 0:
            filename = 'NaiveNet' + str(epoch) + '.pth'
            save_path = os.path.join(save_dir, filename)
            torch.save(model.state_dict(), save_path)


def main(data_path, save_dir):
    f = h5py.File(data_path, 'r')
    # images = np.array(f["input"]).reshape(10000, 1, hw, hw)
    images = np.array(f["input"])
    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])
    label = np.array(f["label"])
    f.close()

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
    args = parser.parse_args()

    data_path = args.data_path
    save_dir = args.save_dir
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    data_size = args.training_size

    main(data_path, save_dir)
