import torch
import torch.nn as nn
import torch.nn.functional as F

from STG_model import StochasticGates

class GVAE(nn.Module):
    def __init__(self, latent_dim=100, gate_type="stg", lamb=1, sigma=0.5, use_gate=True,
                 device="cpu"):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(GVAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

        self.gate_type = gate_type
        self.use_gate  = use_gate
        self.lamb      = lamb
        self.sigma     = sigma

        # -- VAE loss --
        # adding BCELoss to the model-> NOTE: use sum !!
        # see: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        self.BCE = nn.BCELoss(reduction='sum')


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
        self.mu = nn.Linear(64 * 7 * 7, latent_dim)     # 3163->latentDim (default=100)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim) # 3163->latentDim (default=100)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7) # latentDim(default=100)->3163

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  #in:[B,64,7,7], out:[B,64,14,14]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), #in:[B,64,14,14], out:[B,32,28,28]->should be [B,32,29,29]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  #in:[B,64,29,129],out:[B,1,28,28]
            nn.Sigmoid()
        )


    def sample(self,sample_size,mu=None,logvar=None):
        '''
        this function is used to sample new data (a photo/s) from the model

        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)

        z = torch.randn_like(mu).to(self.device)  # [sample_size,latent_dim]

        x = self.upsample(z)      # [sample_size,64*7*7]
        x = x.view(-1, 64, 7, 7)  # [sample_size,64,7,7]
        x = self.decoder(x)       # [sample_size,1,28,28]
        return x


    def z_sample(self, mu, logvar):
        """
        sampling from the dist' trained in the latent space
        :param mu: mean
        :param logvar: var
        :return: return a sampled vector from the dist' (Normal in this model)
        """
        # sampling from ~ N(0,1) with same size as mu
        # torch.randn_like: Returns a tensor with the same size as input that is filled with
        # random numbers from a normal distribution with mean 0 and variance 1
        normal_sapmled = torch.randn_like(mu).to(self.device)

        # The Trick of VAE!
        sigma = torch.sqrt(torch.exp(logvar)).to(self.device)
        mu = mu.to(self.device)

        z = mu + normal_sapmled*sigma

        return z



    def loss(self, x, recon, mu, logvar):
        """
        :param x:       the original data
        :param recon:   the reconstruct data with the model
        :param mu:      mean vector
        :param logvar:  log var vector

        :return: the loss: minimizing 2 terms:
            1. VAE loss:  "(-ELBO)" = -(-BCE-KL)= BCE+KL
            2. gate loss

        for the VAE loss:
            the goal is to max the ELBO, but optimizer always minimize
            this is why we minimize the (-ELBO).

            ELBO in this model is: Eq[logp(x|z)] - KL(q(z|x)||p(z))
                note: (1) binary cross entropy(BCE): -Eq[logp(x|z)]
                      (2) KL(see HW1): KL(q(z|x)||p(z))= 0.5*sum[sigma^2+mu^2-log(sigma^2)-1]
                  -> LOSS = BCE+KL

        for the gate loss:
            depends on the gate type, we calculate the gate loss
        """

        # ---- VAE loss -----
        # could calc like follows, but there is a nicer way I found.
        # bern = torch.sum(x * torch.log(recon) + (1 - x) * torch.log(1 - recon), dim=1)
        BCE = self.BCE(recon, x)  # nicer way to calc the bern'. criterion(input,target)
        KL  = 0.5 * (torch.exp(logvar) + mu ** 2 - logvar - 1).sum()

        """NOTE: in VAE: the goal is to max the ELBO, but 'optimizer' always minimize!
                this is why we minimizing the (-ELBO)."""
        loss_vae = (BCE+KL)

        total_loss = loss_vae

        # --- Gate loss -----
        if self.use_gate is True:
            loss_gate = self.gate.get_loss()
            # print(f"{loss_vae=}")
            # print(f"{loss_gate=}")

            total_loss = loss_vae + self.lamb * loss_gate

        return total_loss


    def forward(self, input):
        # Keep the original input untouched for the VAE reconst' loss!
        x = input

        # ----- Gate ---------
        # pass data to the gate
        if self.use_gate is True:
            batch, C, H, W = x.shape
            x = torch.reshape(x, (batch, -1))
            x = self.gate(x)
            x = torch.reshape(x, (batch,C,H,W))
        # ----------------------------

        # ---- VAE ---------
        # pass data in the encoder model
        latent = self.encoder(x)

        # reshape encoder output to match linear layer for mu and var
        latent = latent.view(-1, 64*7*7)  # Amit: [B,64*7*7]

        # pass linear FNN to get mu and var
        mu_z = self.mu(latent).to(self.device)

        logvar_z = self.logvar(latent).to(self.device)

        # sample random z from dist' of ~N(mu,var) with trick
        z = self.z_sample(mu_z, logvar_z).to(self.device)

        # upsampling from latend_dim to match dim before linear layers (FNN)
        x_hat = self.upsample(z).to(self.device)

        # reshape the data to match the shape of the convs layers
        x_hat = x_hat.view(-1, 64, 7, 7)

        # pass data in decoder - convs layers
        x_hat = self.decoder(x_hat).to(self.device)
        # finish reconstructing the input with the VAE
        # -------------------------------

        # calc loss
        loss_batch = self.loss(input, x_hat, mu_z, logvar_z)


        return loss_batch, x_hat, z




if __name__ == '__main__':
    pass