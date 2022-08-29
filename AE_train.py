import argparse
from asyncore import file_wrapper

import torch, torchvision
from numpy import signedinteger
from torchvision import transforms
import numpy as np
import time
import datetime
import os
import matplotlib.pyplot as plt

from ProjectUtils import choose_cuda
from AE_model import GAE
from ProjectDataset import MNIST_noisy

def train(ae, trainloader, optimizer, epoch, device):
    ae.train()  # set to training mode
    mixed_loss_list = []

    for batch_idx, (data, label) in enumerate(trainloader):
        # print(f"{data.shape=} , {label.shape}")
        start = time.time()
        data = data.to(device=device, dtype=torch.float)

        optimizer.zero_grad()
        mixed_loss, data_reconst, _ = ae(data)
        mixed_loss.backward()
        optimizer.step()

        # in 'loss()' we calc sum , so we divide by the batch size
        mixed_loss_item = mixed_loss.item() / len(data)
        mixed_loss_list.append(mixed_loss_item)
        end = time.time()
        # print(f"epoch:{epoch + 1}/{args.epochs}, batch:{batch_idx}, train_ELBO:{mixed_loss_item}|took:{end-start}[sec]")

    print(" ")
    # calc the mean loss over the trained epoch
    mean_mixed_loss = np.mean(mixed_loss_list)

    return mean_mixed_loss, data_reconst


def test(ae, testloader, epoch, sample_size, device, args):
    ae.eval()  # set to inference mode

    with torch.no_grad():
        mixed_loss_list = []
        for batch_idx, (data, label) in enumerate(testloader):
            start = time.time()
            data = data.to(device=device, dtype=torch.float)

            mixed_loss, data_reconst, _ = ae(data)

            # in 'loss()' we calc sum , so we divide by the batch size
            mixed_loss_item = mixed_loss.item() / len(data)
            mixed_loss_list.append(mixed_loss_item)

            end = time.time()
            # print(f"epoch:{epoch + 1}/{args.epochs}, batch:{batch_idx}, test_ELBO:{mixed_loss_item}|took:{end-start}[sec]")

        # calc the mean loss over the test epoch
        mean_mixed_loss = np.mean(mixed_loss_list)

        return mean_mixed_loss, data_reconst


def main(args):
    device = choose_cuda(args.cuda_num)

    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)),  #dequantization
        transforms.Normalize((0.,), (257./256.,)),  # rescales to [0,1]
    ])

    # ---- Choose the dataset ------
    if args.dataset == 'mnist' or args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=f'{args.dir_root}/data/MNIST',
            train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=f'{args.dir_root}/data/MNIST',
            train=False, download=True, transform=transform)
    elif args.dataset == 'MNIST_images':
        root_dataset   = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
        dir_data_train = f"{root_dataset}/mnist_background_images_test.amat"
        dir_data_test  = f"{root_dataset}/mnist_background_images_train.amat"
        trainset       = MNIST_noisy(dir_data=dir_data_train)
        testset        = MNIST_noisy(dir_data=dir_data_test)
    elif args.dataset == 'MNIST_random':
        root_dataset   = "/dsi/gannot-lab/datasets/Images_datasets/MNIST/NoisyDatset"
        dir_data_train = f"{root_dataset}/mnist_background_random_test.amat"
        dir_data_test  = f"{root_dataset}/mnist_background_random_train.amat"
        trainset       = MNIST_noisy(dir_data=dir_data_train)
        testset        = MNIST_noisy(dir_data=dir_data_test)
    else:
        raise ValueError('Dataset not implemented')
    # ------------------------------

    # --- Create dataloader --------
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=8)
    # ------------------------------

    # --- Create the G-AE model -----
    ae = GAE(latent_dim=args.latent_dim, gate_type=args.gate_type,
               lamb=args.lamb, sigma=args.sigma, use_gate=args.use_gate,
               device=device).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam([
    #     {'params': ae.gate.parameters(), 'lr': args.lr/10},
    #     {'params': ae.encoder.parameters()},
    #     {'params': ae.bottleneck.parameters()},
    #     {'params': ae.upsample.parameters()},
    #     {'params': ae.decoder.parameters()}
    # ], lr=args.lr)

    # lists to hold the train and test loss
    GAE_loss_train = []
    GAE_loss_test  = []

    # -------  start training over the dataset  ---------
    print(f"Start Train on dataset:{args.dataset}")
    start_train = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        loss_epoch_train, data_reconst_train = train(ae, trainloader, optimizer, epoch, device)
        end = time.time()
        print(f"epoch:{epoch+1}/{args.epochs}, ----> loss_train:{loss_epoch_train:.3f}|took:{(end-start):.3f}[sec]")
        GAE_loss_train.append(loss_epoch_train)

        start = time.time()
        loss_epoch_test, data_reconst_test = test(ae, testloader, epoch, args.sample_size, device, args)
        end = time.time()
        print(f"epoch:{epoch + 1}/{args.epochs}, -+-+-> loss_test:{loss_epoch_test:.3f}|took:{(end-start):.3f}[sec]")
        GAE_loss_test.append(loss_epoch_test)

        # save reconstruction data from train and test
        if not os.path.exists(args.dir_reconstruction):
            os.makedirs(args.dir_reconstruction)
        save_path_train = f"{args.dir_reconstruction}/epoch_{epoch + 1}_train.png"
        save_path_test = f"{args.dir_reconstruction}/epoch_{epoch + 1}_test.png"
        torchvision.utils.save_image(torchvision.utils.make_grid(data_reconst_train), save_path_train)
        torchvision.utils.save_image(torchvision.utils.make_grid(data_reconst_test), save_path_test)
    end_train = time.time()
    print(f"Total train and test took:{(end_train-start_train):.3f}[sec]")
    print("--- END of train -----")

    # ---- Saving the model  -----
    if args.save_model is True:
        print("Start Saving model")
        if not os.path.exists(args.dir_models):
            os.makedirs(args.dir_models)
        torch.save({
            'model_state_dict': ae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_epoch': args.epochs,
            'dataset': args.dataset,
            'batch_size': args.batch_size}, # end of parameters-to-be-saved list
            f"{args.dir_models}/model.tar")
        print('Finished saving model - Checkpoint Saved')
    # -----------------------------------------

    # ----- Plot and save the loss: train VS test  -----
    if not os.path.exists(args.dir_fig):
        os.makedirs(args.dir_fig)

    plt.figure()
    plt.plot(GAE_loss_train, linewidth=3, color='blue', label='Train loss')
    plt.plot(GAE_loss_test, linewidth=3, color='orange', label='Test loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.grid(True)

    fig_name = f"loss_TrainVSTest_{args.dataset}_Epochs_{args.epochs}.png"
    plt.savefig(f"{args.dir_fig}/{fig_name}")
    plt.close()
    # ------------------------------

    # ------ Print the gates -----
    if args.use_gate is True:
        # print the largest N mus and their indices
        N = 100
        sorted, indices = torch.sort(ae.gate.mus, descending=True)
        print(f"the top {N=} mus are:")
        print(f"{indices[0:N]=}")
        print(f"{sorted[0:N]=}")

        print(f"the lowest {N=} mus are:")
        print(f"{indices[-N:]=}")
        print(f"{sorted[-N:]=}")
    # -----------------------------


if __name__ == '__main__':
    cuda_num = 3
    # ---------

    choose_dataset = "MNIST_random"  # MNIST, MNIST_images , MNIST_random

    latent_dim   = 100    # AE latent dim

    gate_type    = "stg"  # stg , concrete
    use_gate     = False   # False, True
    lamb         = 1      # gate's regularization
    sigma        = 0.5    # stg sigma value

    epochs       = 300
    batch_size   = 128
    sample_size  = 64
    lr           = 1e-3

    save_model   = True  # False / True

    # ------------------------
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d")  # only date
    save_name = f"{dt_string}_{choose_dataset}" \
                f"_gate_type_{gate_type}_use_gate_{use_gate}_lamb_{lamb}_sigma_{sigma}_latent_dim_{latent_dim}" \
                f"_epochs_{epochs}_batch_size_{batch_size}_lr_{lr}"

    # -----------------------------------------
    dir_root         = "/home/dsi/amiteli/Master/Courses/UnsupervisedLearning/Project/G_AE"
    dir_results      = f"{dir_root}/results"
    dir_model_folder = f"{dir_results}/{save_name}"
    dir_models       = f"{dir_model_folder}/model"
    dir_fig          = f"{dir_model_folder}/figures"
    dir_samples      = f"{dir_fig}/samples"
    dir_reconstruction = f"{dir_fig}/reconstruction"
    # -----------------------------------------

    # -----------------------------------------
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', help='dataset to be modeled.', type=str, default=choose_dataset)
    parser.add_argument('--batch_size', help='number of images in a mini-batch.',type=int, default=batch_size)
    parser.add_argument('--epochs', help='maximum number of iterations.', type=int, default=epochs)
    parser.add_argument('--sample_size', help='number of images to generate.', type=int, default=sample_size)

    parser.add_argument('--latent-dim', help='.', type=int, default=latent_dim)
    parser.add_argument('--gate_type', help='.', type=str, default=gate_type)
    parser.add_argument('--use_gate', help='.', type=bool, default=use_gate)
    parser.add_argument('--lamb', help='gates regularization value', type=float, default=lamb)
    parser.add_argument('--sigma', help='gates sigma value', type=float, default=sigma)

    parser.add_argument('--lr', help='initial learning rate.', type=float, default=lr)

    parser.add_argument('--cuda_num', help='select a cuda device', type=int, default=cuda_num)
    parser.add_argument('--dir_root', help='dir_root', type=str, default=dir_root)
    parser.add_argument('--dir_fig', help='dir_fig', type=str, default=dir_fig)
    parser.add_argument('--dir_models', help='dir_models', type=str, default=dir_models)
    parser.add_argument('--dir_samples', help='dir_samples', type=str, default=dir_samples)
    parser.add_argument('--dir_reconstruction', help='dir_samples', type=str, default=dir_reconstruction)
    parser.add_argument('--save_name', help='save_name', type=str, default=save_name)

    parser.add_argument('--save_model', help='select a cuda device', type=bool, default=save_model)

    args = parser.parse_args()
    main(args)
