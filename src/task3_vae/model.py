import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt

class VAE(nn.Module):

    def __init__(self, d_in:int, d_latent:int, d_hidden_layer:int, device):
        """Initialiize the VAE.

        Args:
            d_in (int): Input dimension
            d_latent (int): Latent dimension.
            d_hidden_layer (int): Number of neurons in the hidden layers of encoder and decoder.
            device: 'cpu' or 'cuda
        """
        super(VAE, self).__init__()
        
        # Set device
        self.device = device
        # TODO: Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
        self.d_in = d_in
        self.d_latent = d_latent
        self.d_hidden_layer = d_hidden_layer
        # TODO: Initialize the encoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden_layer),
            nn.ReLU(), # nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_layer, d_hidden_layer),
            nn.ReLU(), # nn.LeakyReLU(0.2),
        )
        # TODO: Initialize a linear layer for computing the mean (one of the outputs of the encoder)
        self.fc_mu = nn.Linear(d_hidden_layer, d_latent)
        # TODO: Initialize a linear layer for computing the variance (one of the outputs of the encoder)
        self.fc_logvar = nn.Linear(d_hidden_layer, d_latent)
        # TODO: Initialize the decoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden_layer),
            nn.ReLU(), # nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_layer, d_hidden_layer),
            nn.ReLU(), # nn.LeakyReLU(0.2),
            nn.Linear(d_hidden_layer, d_in),
            nn.Sigmoid() # nn.Tanh()
        )
        for p in self.parameters():
            p.requires_grad = True
        self.log_sigma = nn.Parameter(torch.Tensor([0.0]))

    def encode_data(self, x:npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ Forward pass throguh the encoder. 

        Args:
            x (npt.NDArray[np.float64]): Input data

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: mean, log of variance
        """
        # TODO: Implement method!!
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu:npt.NDArray[np.float64], logvar:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Use the reparameterization trick for sampling from a Gaussian distribution.

        Args:
            mu (npt.NDArray[np.float64]): Mean of the Gaussian distribution.
            logvar (npt.NDArray[np.float64]): Log variance of the Gaussian distribution.

        Returns:
            npt.NDArray[np.float64]: Sampled latent vector.
        """
        # TODO: Implement method!!
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z #.cpu().detach().numpy()

    def decode_data(self, z:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Decode latent vectors to reconstruct data.

        Args:
            z (npt.NDArray[np.float64]): Latent vector.

        Returns:
            npt.NDArray[np.float64]: Reconstructed data.
        """
        # TODO: Implement method!!
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(self.device)
        recon_x = self.decoder(z)
        return recon_x


    def generate_data(self, num_samples:int) -> npt.NDArray[np.float64]:
        """ Generate data by sampling and decoding 'num_samples' vectors in the latent space.

        Args:
            num_samples (int): Number of generated data samples.

        Returns:
            npt.NDArray[np.float64]: generated samples.
        """
        # TODO: Implement method!!
        # Hint (You may need to use .to(self.device) for sampling the latent vector!)
        z = torch.randn(num_samples, self.d_latent, requires_grad=True).to(self.device)
        generated_samples = self.decoder(z)
        return generated_samples # .cpu().detach().numpy()
    
    def forward(self, x:npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ Forward pass of the VAE.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: reconstructed data, mean of gaussian distribution (encoder), variance of gaussian distribution (encoder)
        """
        # TODO: Implement method!!
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z) #(torch.tensor(z, dtype=torch.float32, requires_grad=True).to(self.device))
        return recon_x, mu, logvar




