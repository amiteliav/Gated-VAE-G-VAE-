import os

import torch, torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score
from sklearn.manifold import TSNE

# ---
# originally should import as follows, but it is an old version, instead use the mirror code
# #from sklearn.utils.linear_assignment_ import linear_assignment
from linear_assignment_mirror import linear_assignment
# ----------

import seaborn as sns

import time
import pandas as pd
import numpy as np
from torch.cuda import device

from ProjectUtils import choose_cuda
from VAE_model import GVAE
from AE_model import GAE
from ProjectDataset import MNIST_noisy


def load_datasets(dir_root):
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
        root_dataset = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
        dir_data_train = f"{root_dataset}/mnist_background_images_test.amat"
        dir_data_test = f"{root_dataset}/mnist_background_images_train.amat"
        trainset = MNIST_noisy(dir_data=dir_data_train)
        testset = MNIST_noisy(dir_data=dir_data_test)
    elif choose_dataset == 'MNIST_random':
        root_dataset = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
        dir_data_train = f"{root_dataset}/mnist_background_random_test.amat"
        dir_data_test = f"{root_dataset}/mnist_background_random_train.amat"
        trainset = MNIST_noisy(dir_data=dir_data_train)
        testset = MNIST_noisy(dir_data=dir_data_test)
    else:
        raise ValueError(f"Dataset {choose_dataset} is not implemented")

    len_dataset = len(testset)
    print(f"{len_dataset=}")
    cluster_dataloader = torch.utils.data.DataLoader(trainset, batch_size=len_dataset,
                                                     shuffle=True,
                                                     num_workers=8)

    return cluster_dataloader, len_dataset


def retrieve_info(cluster_labels, y_train):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(cluster_labels))):
        index = np.where(cluster_labels == i, 1, 0)
        num = np.bincount(y_train[index == 1]).argmax()
        reference_labels[i] = num

    return reference_labels


def get_best_match_labels(labels_true, labels_kmeans):
    reference_labels = retrieve_info(labels_kmeans, labels_true)

    labels_best_match = np.random.rand(len(labels_kmeans))
    for j in range(len(labels_kmeans)):
        labels_best_match[j] = reference_labels[labels_kmeans[j]]

    return labels_best_match

def get_acc_NMI(labels_true, labels_best_match):
    acc = accuracy_score(labels_true, labels_best_match)
    NMI = normalized_mutual_info_score(labels_true=labels_true, labels_pred=labels_best_match)

    return acc, NMI


def acc_NMI_example():
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])  # true labels
    labels_kmeans = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0, 4, 4, 3, 3])  # output from kmeans

    # get the best match between true labels and kmeans
    labels_best_match = get_best_match_labels(labels_true, labels_kmeans)

    acc, NMI = get_acc_NMI(labels_true, labels_best_match)
    print(f"{acc=} , {NMI=}")


def get_kmeans(data, n_clusters=10, random_state=1):
    # K-means with sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)  # Kmeans object
    kmeans = kmeans.fit(data)      # performe the clustering
    pred   = kmeans.predict(data)  # match between the cluster and label

    return pred

def get_GaussianMixture(data, n_clusters=10):
    gm   = GaussianMixture(n_components=n_clusters, random_state=0).fit(data)
    pred = gm.predict(data)  # match between the cluster and label

    return pred

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return acc


def run_clustering(data, labels_true, choose_dataset=None, clustering_type="kmeans"):
    # Kmeans the data
    data_toClustering = data
    print(f"{data_toClustering.shape=}")

    if clustering_type=="kmeans":
        labels_pred = get_kmeans(data_toClustering, n_clusters=10)
    elif clustering_type == "gmm":
        labels_pred = get_GaussianMixture(data_toClustering, n_clusters=10)
    else:
        print(f"{clustering_type=} NOT supported")

    print(f"--- Clustering result for {clustering_type=} ---")
    #===  for matching and scoring the labels there are some options ===
    # opt1
    labels_best_match = get_best_match_labels(labels_true=labels_true, labels_kmeans=labels_pred)
    acc, NMI = get_acc_NMI(labels_true=labels_true, labels_best_match=labels_best_match)
    print(f'Accuracy = {acc:.3f}%')
    print(f'normalized_mutual_info_score = {NMI:.3f}')

    # opt 2
    hom_score = homogeneity_score(labels_true=labels_true, labels_pred=labels_pred)
    print(f"{hom_score=}")

    # opt 3
    cluster_acc_result = cluster_acc(y_true=labels_true, y_pred=labels_pred)
    print(f"{cluster_acc_result=}")
    #######################



def base_clustering(data_input, labels_true, flag_tsne = False, choose_dataset=None):
    print(f"===== {choose_dataset} - base_clustering  =====")

    # norm' the data
    data = (data_input - np.mean(data_input)) / np.std(data_input)

    # Plot TSNE
    if flag_tsne is True:
        note = "base clustering"
        plot_tsne(data=data, labels_true=labels_true, epoch=0, choose_dataset=choose_dataset, note=note)

    # Shuffling the dataset  # TODO: should i shuffle? also the labels!
    # rand_indices  = np.random.permutation(data.shape[0])
    # data          = data[rand_indices]

    # Run Kmeans and Plot clustering results
    run_clustering(data, labels_true)

    # Run GMM and Plot clustering results
    run_clustering(data, labels_true, clustering_type="gmm")


def VAE_gated_clustering(model, epoch, data_input, labels_true, N_features = 100, flag_tsne = False, use_gates_as_weight=True,
                         device="cpu", choose_dataset=None):
    print(f"===== {choose_dataset} - VAE_clustering - using only feature selection - weighted={use_gates_as_weight} - {epoch=} =====")
    model.eval()

    # use the model to get a feature selection
    sorted, indices = torch.sort(model.gate.mus, descending=True)
    sorted          = sorted[0:N_features].detach().cpu().numpy()
    indices         = indices[0:N_features].detach().cpu().numpy()

    # # Print top N gates
    # print(f"the top {N_features=} mus are:")
    # print(f"{indices[0:N_features]=}")
    # print(f"{sorted[0:N_features]=}")

    # ==== Use the gates as feature selection, and run Kmeans  =====
    #  feature selection
    data  = data_input[:, indices]
    if use_gates_as_weight is True:
        data  = data*sorted      # use the gates as weights

    # norm' the data
    data = (data - np.mean(data)) / np.std(data)

    # Plot TSNE
    if flag_tsne is True:
        note = f"VAE_clustering - using only feature selection - weighted={use_gates_as_weight}\n"
        plot_tsne(data=data, labels_true=labels_true, epoch=epoch,
                  choose_dataset=choose_dataset, note=note)


    # # shuffling  # TODO: should i shuffle? also the labels!
    # rand_indices  = np.random.permutation(data.shape[0])
    # data          = data[rand_indices]

    # Run Kmeans and Plot clustering results
    run_clustering(data, labels_true)

    # Run GMM and Plot clustering results
    run_clustering(data, labels_true, clustering_type="gmm")

def VAE_z_latent_clustering(model, epoch, data_input, labels_true, flag_tsne = False, device="cpu",
                            learning_rate='auto', choose_dataset=None):
    print(f"===== {choose_dataset} - VAE_clustering - using GVAE latent space - {epoch=} =====")
    # ----- Using the VAE latent spaces for clustering  ---------
    model.to(device)
    model.eval()
    batch       = data_input.shape[0]
    data        = data_input

    # norm' the data: with GVAE I choose not to norm the data, to use the same way as in train

    data_tensor = torch.from_numpy(data.reshape(batch,1,28,28)).to(device)
    _,_, data_z = model.forward(data_tensor.float())
    data_z = data_z.detach().cpu().numpy()

    # Plot TSNE
    if flag_tsne is True:
        note = f"VAE_z_latent_clustering\n"
        plot_tsne(data=data_z, labels_true=labels_true, epoch=epoch,
                  choose_dataset=choose_dataset, note=note)

    # Run Kmeans and Plot clustering results
    run_clustering(data_z, labels_true)

    # Run GMM and Plot clustering results
    run_clustering(data, labels_true, clustering_type="gmm")


def plot_tsne(data, labels_true,epoch, N = None, learning_rate='auto', choose_dataset=None, note=None):
    if N is not None:  # Run over only a subset of the data
        data        = data[:N,:]
        labels_true = labels_true[:N]
    init = "random"  # "random" , "pca"
    tsne = TSNE(n_components=2, perplexity=50, n_iter=300, random_state=0,
                learning_rate=learning_rate, init=init, n_jobs=-1, verbose=0,
                early_exaggeration=12)
    tsne_result = tsne.fit_transform(data)

    # Plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1],
                         c=labels_true, s=15, cmap='tab10')

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    ax.add_artist(legend1)
    if choose_dataset is not None:
        plt.title(f"{choose_dataset} - {note} - TSNE for {epoch=}")
    else:
        plt.title(f"TSNE for {epoch=} , - {note}")
    plt.show()
    plt.close()


def plot_project_results(model_type, choose_dataset, epochs, N_features, n_samples, gated_model, device, flag_tsne=False):
    dir_root   = "/home/dsi/amiteli/Master/Courses/UnsupervisedLearning/Project"
    if model_type=="VAE":
        dir_models = f"{dir_root}/G_VAE/results"
    elif model_type=="AE":
        dir_models = f"{dir_root}/G_AE/results"
    else:
        print(f"ERROR- {model_type=} NOT suppoerted")
    # ------------------------------------

    # ---- loading the dataset -------
    cluster_dataloader, len_dataset = load_datasets(dir_root)
    data_input, labels_true = next(iter(cluster_dataloader))
    data_input = data_input.cpu().numpy().reshape(len_dataset, -1)
    labels_true = labels_true.cpu().numpy()

    print(f"dataset shape:{data_input.shape}")
    data = data_input[:n_samples, :]
    labels_true = labels_true[:n_samples].astype(int)
    print(f"{data.shape=}")
    print(f"{labels_true.shape=}")
    # ---------------------------------

    # Kmeans with full size dataset - with not feature selection etc.
    base_clustering(data_input=data, labels_true=labels_true, flag_tsne=flag_tsne,
                    choose_dataset=choose_dataset)

    print("==========================")

    sorted_gates_list = []  # save the sorted gates from the model for the different epochs

    # #  ------- Path to the trained GVAE --------------
    for epoch in epochs:
        if model_type == "VAE":
            str_folder  = f"20220823_{choose_dataset}_gate_type_stg_use_gate_{gated_model}_lamb_1.1_sigma_0.5_latent_dim_100_epochs_{epoch}_batch_size_128_lr_0.001"
        elif model_type == "AE":
            str_folder = f"20220828_{choose_dataset}_gate_type_stg_use_gate_{gated_model}_lamb_1_sigma_0.5_latent_dim_100_epochs_{epoch}_batch_size_128_lr_0.001"
        else:
            print(f"ERROR- {model_type=} NOT suppoerted")
        path_model = f"{dir_models}/{str_folder}/model/model.tar"

        # -------- Create G-VAE /G-AE model  -------------
        print(f"loading model for {epoch=}")
        if model_type == "VAE":
            model = GVAE(latent_dim=100, gate_type="stg", lamb=1.1, sigma=0.5, use_gate=gated_model, device=device).to(device)
        elif model_type == "AE":
            model = GAE(latent_dim=100, gate_type="stg", lamb=1, sigma=0.5, use_gate=gated_model, device=device).to(device)
        else:
            print(f"ERROR- {model_type=} NOT suppoerted")
        checkpoint = torch.load(path_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"model loaded! from path:{path_model}")
        model.eval()
        # ---------------------------------

        if gated_model is True:
            # get the gates
            sorted, indices = torch.sort(model.gate.mus, descending=True)
            sorted_gates_list.append(sorted.detach().cpu().numpy())
            # --------------------------

            # Kmeans over the selection features - weighted=False
            VAE_gated_clustering(model=model, epoch=epoch, data_input=data, labels_true=labels_true, N_features=N_features,
                                 flag_tsne=flag_tsne, use_gates_as_weight=False, device=device, choose_dataset=choose_dataset)

            print("==========================")

            # Kmeans over the selection features - weighted=True
            VAE_gated_clustering(model=model, epoch=epoch, data_input=data, labels_true=labels_true, N_features=N_features,
                                 flag_tsne=flag_tsne, use_gates_as_weight=True, choose_dataset=choose_dataset)

            print("==========================")

        # Kmeans over the GVAE laten space
        tnse_lr = 'auto'
        VAE_z_latent_clustering(model=model, epoch=epoch, data_input=data, labels_true=labels_true,
                                flag_tsne=flag_tsne, device=device, learning_rate=tnse_lr, choose_dataset=choose_dataset)



        print("==========================")

    if gated_model is True:
        # Plot all the sorted gates, for all the different epochs
        fig, ax = plt.subplots()
        for i in range(len(epochs)):
            plt.plot(sorted_gates_list[i], label=f"epoch {epochs[i]}")
        plt.legend()
        plt.title(f"G-VAE - Gates values, Dataset: {choose_dataset}")
        plt.show()
        plt.close()


def plot_AE_resultes():
    dir_root = "/home/dsi/amiteli/Master/Courses/UnsupervisedLearning/Project"
    dir_models = f"{dir_root}/G_AE/results"
    # ------------------------------------

    flag_tsne = True

    # ---- loading the dataset -------
    cluster_dataloader, len_dataset = load_datasets(dir_root)
    data_input, labels_true = next(iter(cluster_dataloader))
    data_input = data_input.cpu().numpy().reshape(len_dataset, -1)
    labels_true = labels_true.cpu().numpy()

    print(f"dataset shape:{data_input.shape}")
    data = data_input[:n_samples, :]
    labels_true = labels_true[:n_samples].astype(int)
    print(f"{data.shape=}")
    print(f"{labels_true.shape=}")
    # ---------------------------------

    sorted_gates_list = []  # save the sorted gates from the model for the different epochs
    # #  ------- Path to the trained GVAE --------------
    for epoch in epochs:
        # 20220828_MNIST_gate_type_stg_use_gate_False_lamb_1_sigma_0.5_latent_dim_100_epochs_300_batch_size_128_lr_0.001
        str_folder = f"20220828_{choose_dataset}_gate_type_stg_use_gate_{gated_model}_lamb_1_sigma_0.5_latent_dim_100_epochs_{epoch}_batch_size_128_lr_0.001"
        path_model = f"{dir_models}/{str_folder}/model/model.tar"

        # -------- Create G-AE model  -------------
        print(f"loading model for {epoch=}")
        model = GAE(latent_dim=100, gate_type="stg", lamb=1, sigma=0.5, use_gate=gated_model, device=device).to(device)
        checkpoint = torch.load(path_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # ---------------------------------

        if gated_model is True:
            # get the GAE gates
            sorted, indices = torch.sort(model.gate.mus, descending=True)
            sorted_gates_list.append(sorted.detach().cpu().numpy())
            # --------------------------

            # Kmeans over the selection features - weighted=False
            # Note that 'VAE_gated_clustering()' can be used both for AE and VAE
            VAE_gated_clustering(model=model, epoch=epoch, data_input=data, labels_true=labels_true,
                                 N_features=N_features,
                                 flag_tsne=flag_tsne, use_gates_as_weight=False, device=device,
                                 choose_dataset=choose_dataset)

            print("==========================")

            # Kmeans over the selection features - weighted=True
            # Note that 'VAE_gated_clustering()' can be used both for AE and VAE
            VAE_gated_clustering(model=model, epoch=epoch, data_input=data, labels_true=labels_true,
                                 N_features=N_features,
                                 flag_tsne=flag_tsne, use_gates_as_weight=True, choose_dataset=choose_dataset)

            print("==========================")

        # Kmeans over the GAE laten space
        # Note that 'VAE_z_latent_clustering()' can be used both for AE and VAE
        tnse_lr = 'auto'
        VAE_z_latent_clustering(model=model, epoch=epoch, data_input=data, labels_true=labels_true,
                                flag_tsne=flag_tsne, device=device, learning_rate=tnse_lr,
                                choose_dataset=choose_dataset)

        print("==========================")

    if gated_model is True:
        # Plot all the sorted gates, for all the different epochs
        fig, ax = plt.subplots()
        for i in range(len(epochs)):
            plt.plot(sorted_gates_list[i], label=f"epoch {epochs[i]}")
        plt.legend()
        plt.title(f"G-VAE - Gates values, Dataset: {choose_dataset}")
        plt.show()
        plt.close()


if __name__ == '__main__':
    cuda_num = "cpu"
    device = choose_cuda(cuda_num)

    choose_dataset  = "MNIST_random"  # MNIST, MNIST_images , MNIST_random
    model_type      = "VAE"     # VAE or AE
    flag_tsne       = False
    gated_model     = True    #
    epochs          = [10,30,50,100,300]    # [10,30,50,100,300]
    N_features      = 40      # number of features to select, usually 100
    n_samples       = None     # run over subset of the dataset, None or int number
    plot_project_results(model_type, choose_dataset, epochs, N_features, n_samples, gated_model, device, flag_tsne)








