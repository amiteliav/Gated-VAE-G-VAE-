import torch
import torch.nn as nn
import torch.nn.functional as F

from STG_model import StochasticGates

class GAE(nn.Module):
    def __init__(self, latent_dim=100, gate_type="stg", lamb=1, sigma=0.5, use_gate=True,
                 device="cpu"):
        """
        Initialize a Gated-AE:
        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(GAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

        self.gate_type = gate_type
        self.use_gate  = use_gate
        self.lamb      = lamb
        self.sigma     = sigma

        # -- VAE loss --
        # adding BCELoss to the model-> NOTE: use sum !!
        # see: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        self.loss_AE = nn.MSELoss()


        # ======  The NN model ========
        # Gate
        if self.use_gate is True:
            if self.gate_type == "stg":
                gate_size = 784  # MNIST are 28*28=784 pixels
                self.gate = StochasticGates(size=gate_size, sigma=self.sigma, gate_init=None)
            elif self.gate_type == "concrete":
                pass # TODO: later...
            else:
                print(f"!ERROR! gate_type={self.gate_type} NOT SUPPORTED ")
        else:
            self.gate = None

        # VAE
        self.encoder = nn.Sequential(
            # Conv2d: in_channels,out_channels,kernel_size,stride,padding,
            nn.Conv2d(1, 32, 4, 1, 2), # in:[B,1,28,28], out:[B,32,29,29]
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # in:[B,32,29,29] out:[B,32,14,14]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # in:[B,32,14,14] out:[B,64,7,7]
        )

        # 64*7*7=3163
        self.bottleneck = nn.Linear(64 * 7 * 7, latent_dim)     # 3163->latentDim (default=100)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7) # latentDim(default=100)->3163

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  #in:[B,64,7,7], out:[B,64,14,14]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), #in:[B,64,14,14], out:[B,32,28,28]->should be [B,32,29,29]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  #in:[B,64,29,129],out:[B,1,28,28]
            nn.Sigmoid()
        )

    def loss(self, x, recon):
        """
        :param x:     the original data
        :param recon: the reconstruct data with the model

        :return: the loss: minimizing 2 terms:
            1. AE loss: reconstruction loss: MSE
            2. gate loss
        """
        # ---- VE loss -----
        batch_loss_AE = self.loss_AE(x, recon)
        total_loss = batch_loss_AE

        # --- Gate loss: if gate is used, get it's loss -----
        if self.use_gate is True:
            loss_gate = self.gate.get_loss()
            print(f"{batch_loss_AE.item()=}")
            print(f"{loss_gate.item()=}")
            total_loss = batch_loss_AE + self.lamb * loss_gate


        return total_loss


    def forward(self, input):
        # Keep the original input untouched for the AE reconst' loss!
        x = input

        # ----- Gate ---------
        # pass data to the gate
        if self.use_gate is True:
            batch, C, H, W = x.shape
            x = torch.reshape(x, (batch, -1))
            x = self.gate(x)
            x = torch.reshape(x, (batch,C,H,W))
        # ----------------------------

        # ---- AE ---------
        # pass data in the encoder model
        x_encoded = self.encoder(x)

        # reshape encoder output to match linear layer
        x_encoded = x_encoded.view(-1, 64*7*7)  # Amit: [B,64*7*7]

        # pass linear FNN to get the latent space representation
        latent = self.bottleneck(x_encoded).to(self.device)

        # upsampling from latend_dim to match dim before linear layers (FNN)
        x_hat = self.upsample(latent).to(self.device)

        # reshape the data to match the shape of the convs layers
        x_hat = x_hat.view(-1, 64, 7, 7)

        # pass data in decoder - convs layers
        x_hat = self.decoder(x_hat).to(self.device)
        # finish reconstructing the input with the AE
        # -------------------------------

        # calc loss
        loss_batch = self.loss(input, x_hat)


        return loss_batch, x_hat, latent




if __name__ == '__main__':
    AE = GAE()

    input = torch.randn((5,1,28,28))
    loss_batch, x_hat, latent = AE(input)

    print(f"{input.shape=} , {x_hat.shape=}, {latent.shape=}")