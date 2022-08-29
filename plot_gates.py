import torch, torchvision
import matplotlib.pyplot as plt
from torchvision import transforms


import sklearn
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.manifold import TSNE

import seaborn as sns

import time
import pandas as pd
import numpy as np
from torch.cuda import device

from ProjectUtils import choose_cuda
from VAE_model import GVAE
from ProjectDataset import MNIST_noisy



def plot_gate_values(model, choose_dataset, epochs):
    # use the model to get a feature selection
    sorted, indices = torch.sort(model.gate.mus, descending=True)

    # Plot
    fig, ax = plt.subplots()
    plt.plot(sorted.detach().cpu().numpy())
    plt.title(f"sorted gates values, dataset:{choose_dataset}, {epochs=}")
    plt.show()
    plt.close()


def plot_gates_as_image(model, choose_dataset, epochs, data, N_features=100):
    # use the model to get a feature selection
    gates  = model.gate.mus
    sorted, indices = torch.sort(gates, descending=True)

    gates = gates.detach().cpu().numpy()

    sorted  = sorted[0:N_features + 1].detach().cpu().numpy()
    indices = indices[0:N_features + 1].detach().cpu().numpy()

    min_gate = np.min(sorted)
    gate_2_img = np.where(gates>min_gate,1,0)
    alpha      = np.where(gates>min_gate,1,0).astype(float)  # set the gates' plot transparency

    # sort the gates, np only sort ascending, so use this -(-) trick
    gate_2_img_descending = -np.sort(-gate_2_img)
    # Plot
    fig, ax = plt.subplots()
    plt.plot(gate_2_img_descending)
    plt.title(f"Plot the Gates as image - dataset:{choose_dataset}, {epochs=}")
    plt.show()
    plt.close()
    # ------------------

    gate_2_img = gate_2_img.reshape(28,28)
    batch = data_input.shape[0]
    data_plot  = data_input.reshape(batch, 28, 28)
    alpha      = alpha.reshape(28,28)

    # Plot
    fig, ax = plt.subplots()
    plt.imshow(gate_2_img)
    plt.title(f"Plot the Gates as image - dataset:{choose_dataset}, {epochs=}")
    plt.show()
    plt.close()

    # Plot
    fig = plt.figure(figsize=(6,3))
    fig.suptitle(f"Gates over the dataset samples\n dataset:{choose_dataset}, {epochs=}", fontsize=14)
    for i in range(batch):
        fig.add_subplot(1, batch, i+1)
        plt.imshow(data_plot[i], cmap='gray')
        plt.imshow(gate_2_img, alpha=alpha)
    plt.show()
    plt.close()



if __name__ == '__main__':
    cuda_num    = 0  # "cpu"  # 0
    device      = choose_cuda(cuda_num)
    dir_root    = "/home/dsi/amiteli/Master/Courses/UnsupervisedLearning/Project"

    N_features       = 40
    choose_dataset   = "MNIST"  # MNIST, MNIST_images , MNIST_random
    epochs           = 50   # choose which epoch-run to use


    path_model = f"{dir_root}/G_VAE/results/" \
                 f"20220823_{choose_dataset}_gate_type_stg_use_gate_True_lamb_1.1_sigma_0.5_latent_dim_100_epochs_{epochs}_batch_size_128_lr_0.001/" \
                 f"model/model.tar"
    # =============================
    # =============================

    # ---- loading the dataset -------
    dir_data_mnist = f"{dir_root}/G_VAE"
    if choose_dataset == 'mnist' or choose_dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.)),  # dequantization
            transforms.Normalize((0.,), (257. / 256.,)),  # rescales to [0,1]
        ])
        trainset = torchvision.datasets.MNIST(root=f'{dir_data_mnist}/data/MNIST',
            train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=f'{dir_data_mnist}/data/MNIST',
            train=False, download=True, transform=transform)
    elif choose_dataset == 'MNIST_images':
        root_dataset   = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
        dir_data_train = f"{root_dataset}/mnist_background_images_test.amat"
        dir_data_test  = f"{root_dataset}/mnist_background_images_train.amat"
        trainset       = MNIST_noisy(dir_data=dir_data_train)
        testset        = MNIST_noisy(dir_data=dir_data_test)
    elif choose_dataset == 'MNIST_random':
        root_dataset   = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
        dir_data_train = f"{root_dataset}/mnist_background_random_test.amat"
        dir_data_test  = f"{root_dataset}/mnist_background_random_train.amat"
        trainset       = MNIST_noisy(dir_data=dir_data_train)
        testset        = MNIST_noisy(dir_data=dir_data_test)
    else:
        raise ValueError('Dataset is not implemented')

    batch_size = 10
    cluster_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ----------- Load data -------------
    data_input, labels_true = next(iter(cluster_dataloader))
    data_input  = data_input.cpu().numpy().reshape(batch_size,-1)
    labels_true = labels_true.cpu().numpy()

    # --------------------------------------
    # Create G-VAE model
    model = GVAE(latent_dim=100, gate_type="stg", lamb=1.1, sigma=0.5,
                 use_gate=True, device=device).to(device)
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    # Plot the values of the gates
    plot_gate_values(model, choose_dataset, epochs)

    # Plot the gates as images
    plot_gates_as_image(model, choose_dataset, epochs, data=data_input,  N_features=N_features)











