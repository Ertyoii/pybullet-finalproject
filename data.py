import sys
import h5py as h5py
import torch
from multiprocessing import Pool, cpu_count
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


def load_data(data_filename):
    f = h5py.File(data_filename, 'r')
    images = np.array(f["input"])
    images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))

    labels = np.array(f["label"])
    f.close()

    return images, labels


def worker(start, end):
    temp_train = []
    temp_label = []
    while start < end:
        success, seg_img, xya = build_and_grasp(view_matrix, projection_matrix)
        if success:
            # print(start)
            temp_train.append(seg_img)
            temp_label.append(xya)
            start += 1

    return temp_train, temp_label


def build_dataset_parallel():
    max_threads = cpu_count()

    pool = Pool(processes=max_threads)
    train, label = zip(*pool.starmap(worker, [(int(volume/max_threads*start), int(volume/max_threads*(start+1))) for start in range(max_threads)]))
    pool.close()

    train = np.array(train)
    train = train.reshape((train.shape[0]*train.shape[1], train.shape[2], train.shape[3]))
    label = np.array(label)
    label = label.reshape((label.shape[0]*label.shape[1], 3))

    h5 = h5py.File(data_filename, "w")

    h5.create_dataset('input', data=np.array(train))
    h5.create_dataset('label', data=np.array(label))


def build_dataset():
    count = 0
    train = []
    label = []
    while count < volume:
        success, seg_img, xya = build_and_grasp(view_matrix, projection_matrix)
        if success:
            # print(count)
            train.append(seg_img)
            label.append(xya)
            count += 1

    train = np.array(train)
    label = np.array(label)
    # print(train.shape)
    # print(label.shape)

    h5 = h5py.File(data_filename, "w")

    h5.create_dataset('input', data=train)
    h5.create_dataset('label', data=label)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} data_filename".format(sys.argv[0]))
        sys.exit(1)

    if not sys.argv[1].endswith(".h5"):
        print("Input must be in h5 format")
        sys.exit(1)

    view_matrix, projection_matrix = init(0)
    volume = cpu_count() * 50
    data_filename = sys.argv[1]

    build_dataset_parallel()
    print("Dataset building completed.")
