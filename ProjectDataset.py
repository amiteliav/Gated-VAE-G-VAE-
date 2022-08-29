import os
import time
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import Dataset

from ProjectUtils import choose_cuda



class MNIST_noisy(Dataset):
    def __init__(self, dir_data):
        """
        dir_data        : path to the data file, for all 4 datasets in format of "mnist_XXX.amat"
        """
        self.dir_data       = dir_data
        self.n_classes      = 10

        self.full_data      = np.loadtxt(self.dir_data)
        # print(f"{self.full_data.shape=}")

        self.dataset_len    = self.full_data.shape[0]
        # print(f"{self.dataset_len=}")

    def __len__(self):
        return self.dataset_len


    def __getitem__(self, idx):
        """
        This is a MNIST dataset, but it is saved as follows:
        if dataset len is N -> [N,28*28+1]=[N,784+1]=[N,785]
        where the [0:-1] is the image, and the last index is the label
        """
        data = self.full_data[idx, :-1].reshape([28, 28]).T
        data = torch.from_numpy(data)
        data = data.unsqueeze(0)

        label = self.full_data[idx, -1]

        return data, label


if __name__ == '__main__':
    cuda_num = 2
    device = choose_cuda(cuda_num)

    ########
    root_dataset   = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
    # dir_data     = f"{root_dataset}/mnist_background_images_test.amat"
    # dir_data     = f"{root_dataset}/mnist_background_random_test.amat"
    # dir_data     = f"{root_dataset}/mnist_background_images_train.amat"
    dir_data     = f"{root_dataset}/mnist_background_random_train.amat"

    batch_size     = 5
    shuffle_flag   = True
    num_workers    = 8
    mnist_noisy_dataset        = MNIST_noisy(dir_data=dir_data)
    loader_mnist_random_train = torch.utils.data.DataLoader(mnist_noisy_dataset, batch_size=batch_size,
                                                            shuffle=shuffle_flag,
                                                            num_workers=num_workers)
    ########



    ########
    # Plot some examples from the data
    batch_data=None
    for batch_idx, (data, labels) in enumerate(loader_mnist_random_train):
        # print(f"{data.shape=} , {labels.shape}")
        batch_data = data
        batch_labels = labels
        break
    n_plot = min(batch_size, 10)
    fig, axes = plt.subplots(1, n_plot)
    for i in range(n_plot):
        axes[i].imshow(batch_data[i].squeeze(0), cmap='gray')
    fig.show()
    ########
