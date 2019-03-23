import torch
from torch.nn import functional as F
from torch import optim
import torch.nn as nn


class FactorKLoss:
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    is_mutual_info : bool
        True : includes the mutual information term in the loss
        False : removes mutual information

    References :
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, gamma=10.,
                 disc_kwargs=dict(neg_slope=0.2, latent_dim=10, hidden_units=1000),
                 optim_kwargs=dict(lr=5e-4, betas=(0.5, 0.9))):
        super().__init__()
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, data, optimizer, model):
        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, mu, logsigma, latent_sample1 = model(data1, is_sample=True)
        rec_loss = F.mse_loss(recon_batch, data1, size_average=False)
        kl_loss = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        d_z = self.discriminator(latent_sample1)

        # clamping to 0 because TC cannot be negative : TEST
        tc_loss = (F.logsigmoid(d_z) - F.logsigmoid(1 - d_z)).clamp(0).mean()

        vae_loss = rec_loss + kl_loss + self.gamma * tc_loss

        # Run VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Discriminator Loss
        # Get second sample of latent distribution
        mu, logsigma = model.encoder(data2)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        latent_sample2 = eps.mul(sigma).add_(mu)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)
        # Calculate total correlation loss
        d_tc_loss = - (0.5 * (F.logsigmoid(d_z) + F.logsigmoid(1 - d_z_perm))).mean()

        # Run discriminator optimizer
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        self.optimizer_d.step()

        return vae_loss


class Discriminator(nn.Module):
    def __init__(self,
                 neg_slope=0.2,
                 latent_dim=10,
                 hidden_units=1000):
        """Discriminator proposed in [1].

        Parameters
        ----------
        neg_slope: float
            Hyperparameter for the Leaky ReLu

        latent_dim : int
            Dimensionality of latent variables.

        hidden_units: int
            Number of hidden units in the MLP

        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits

        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).

        """
        super(Discriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        out_units = 1

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)

        return z


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm
