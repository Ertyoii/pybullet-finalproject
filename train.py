import os
import csv
import argparse
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import *
from model import *

parser = argparse.ArgumentParser(
    description='Hyperparameters and locations'
)
parser.add_argument('--train_path', type=str, required=True, help='path of h5 train data file')
parser.add_argument('--eval_path', type=str, required=True, help='path of h5 eval data file')
parser.add_argument('--ckpt_dir', type=str, required=True, help='path to save checkpoints')
parser.add_argument('--log_dir', type=str, required=True, help='path to save logs of train and eval loss')
parser.add_argument('--learning_rate', '--lr', type=float, required=True, help='learning rate')
parser.add_argument('--epochs', '-ep', type=int, default=1000, help='epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=64, help='batch size')
parser.add_argument('--training_size', type=int, default=256, help='size of training data')


def train(train_loader, eval_loader, use_gpu, optimizer, model, criterion):
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

        if epoch % 10 == 0:
            train_loss = loss.item()
            val_loss = val(eval_loader, use_gpu, model, criterion)
            print("epoch {}, Training    Loss {}".format(epoch, train_loss))
            print("epoch {}, Evaluation  Loss {}".format(epoch, val_loss))
            f = open(os.path.join(log_dir, "loss.csv"), "a")
            writer = csv.writer(f)
            writer.writerow([train_loss, val_loss])
            f.close()

        if epoch % 100 == 0:
            filename = 'NaiveNet' + str(epoch) + '.pth'
            save_path = os.path.join(ckpt_dir, filename)
            torch.save(model.state_dict(), save_path)


def val(eval_loader, use_gpu, model, criterion):
    loss = []
    for _, batch in enumerate(eval_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['l'].cuda())
        else:
            inputs, labels = Variable(batch['X']), Variable(batch['l'])

        outputs = model(inputs)

        if use_gpu:
            outputs = outputs.cuda()
            labels = labels.cuda()

        loss.append(criterion(outputs, labels).item())
    return np.sum(np.array(loss))


def main():
    # prepare training and evaluating data
    f = h5py.File(train_path, 'r')
    images = np.array(f["input"])
    images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))
    label = np.array(f["label"])
    f.close()

    train_data = RobotDataset(images[:data_size, :, :, :], label[:data_size, :])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    f = h5py.File(eval_path, 'r')
    images = np.array(f["input"])
    images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))
    label = np.array(f["label"])
    f.close()

    eval_data = RobotDataset(images[:data_size, :, :, :], label[:data_size, :])
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # prepare model
    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model.cuda()

    train(train_loader, eval_loader, use_gpu, optimizer, model, criterion)


if __name__ == "__main__":
    args = parser.parse_args()

    train_path = args.train_path
    eval_path = args.eval_path
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    data_size = args.training_size

    main()
