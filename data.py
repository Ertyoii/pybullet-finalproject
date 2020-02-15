import os
import cv2
import numpy as np
import h5py as h5py

import torch
from torch.utils.data import Dataset


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


def load_data():
    f = h5py.File("data.h5", 'r')
    input = np.array(f["input"]).reshape(10000, 1, 256, 256)
    label = np.array(f["label"])
    f.close()

    return input, label


def build_h5():
    label = np.genfromtxt("label.csv", delimiter=",")
    train = []
    for file in os.listdir("./data"):
        if file.endswith(".jpeg"):
            filename = os.path.join("./data", file)
            train.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    train = np.array(train)
    h5 = h5py.File('data.h5', 'w')
    print(label.shape)
    print(train.shape)

    h5.create_dataset('input', data=train)
    h5.create_dataset('label', data=label)


if __name__ == "__main__":
    build_h5()
