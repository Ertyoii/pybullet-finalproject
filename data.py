import csv
import os
import sys

import cv2
import h5py as h5py
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from grasp_robot import *


class RobotDataset(Dataset):
    def __init__(self, image, target):
        self.image = torch.from_numpy(image).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx):
        x = self.image[idx]
        label = self.target[idx]
        sample = {'X': x, 'l': label}
        return sample

    def __len__(self):
        return len(self.image)


def test_image_in_dataset():
    input, _ = load_data()
    img = input[2].reshape(256, 256)
    plt.imshow(img)
    plt.show()


def load_data():
    f = h5py.File("data.h5", 'r')
    images = np.array(f["input"]).reshape(10000, 1, 256, 256)
    labels = np.array(f["label"])
    f.close()

    return images, labels


def build_h5(data_dir):
    label = np.genfromtxt("label.csv", delimiter=",")
    train = []
    for file in os.listdir(data_dir):
        if file.endswith(".jpeg"):
            filename = os.path.join(data_dir, file)
            train.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

    train = np.array(train)
    h5 = h5py.File('data.h5', 'w')
    print(label.shape)
    print(train.shape)

    h5.create_dataset('input', data=train)
    h5.create_dataset('label', data=label)


def build_dataset(data_dir, view_matrix, projection_matrix, n):
    count = 0
    f = open("label.csv", "a")
    writer = csv.writer(f)
    while count < n:

        success, seg_img, xya = build_and_grasp(view_matrix, projection_matrix)
        if success:
            print(count)
            filename = os.path.join(data_dir, str(count) + ".jpeg")
            cv2.imwrite(filename, seg_img)
            writer.writerow(xya)
            count += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} data_dir".format(sys.argv[0]))
        sys.exit(1)

    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    view_matrix, projection_matrix = init(0)

    build_dataset(data_dir, view_matrix, projection_matrix, 100000)
    print("Dataset built success.")

    build_h5(data_dir)
