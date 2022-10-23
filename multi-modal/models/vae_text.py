import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE

from datamodules.unimodal import alphabet
from datamodules.unimodal import Text as TextDataset

# Constants
dim = 64
noise = 1e-15


class FeatureEncText(nn.Module):
    def __init__(self):
        super(FeatureEncText, self).__init__()
        num_features = len(alphabet)

        self.conv1 = nn.Conv1d(num_features, 2*dim, kernel_size=1)
        self.conv2 = nn.Conv1d(2*dim, 2*dim, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv5 = nn.Conv1d(2*dim, 2*dim, kernel_size=4, stride=2, padding=0, dilation=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(-2, -1)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        h = out.flatten(start_dim=1)
        return h


class EncoderText(nn.Module):
    def __init__(self, flags):
        super(EncoderText, self).__init__()
        self.flags = flags
        self.text_feature_enc = FeatureEncText()

        self.latent_mu = nn.Linear(in_features=2*dim, out_features=flags.latent_dim, bias=True)
        self.latent_logvar = nn.Linear(in_features=2*dim, out_features=flags.latent_dim, bias=True)

    def forward(self, x):
        h = self.text_feature_enc(x)

        latent_space_mu = self.latent_mu(h)
        latent_space_logscale = self.latent_logvar(h)
        scale = F.softplus(latent_space_logscale) + noise
        return torch.stack((latent_space_mu, scale), dim=0)


class DecoderText(nn.Module):
    def __init__(self, flags):
        super(DecoderText, self).__init__()
        self.flags = flags
        self.linear = nn.Linear(flags.latent_dim, 2*dim)
        self.conv1 = nn.ConvTranspose1d(2*dim, 2*dim, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv2 = nn.ConvTranspose1d(2*dim, 2*dim, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv_last = nn.Conv1d(2*dim, len(alphabet), kernel_size=1)
        self.relu = nn.ReLU()
        self.out_act = nn.Softmax(dim=-2)

    def forward(self, z):
        z = self.linear(z)
        x_hat = z.flatten(end_dim=-2).unsqueeze(-1)
        x_hat = self.conv1(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv_last(x_hat)
        pi = self.out_act(x_hat)
        pi = pi.transpose(-2, -1)
        pi = pi.view(z.size()[:-1] + pi.size()[1:])
        return pi.unsqueeze(0)


class Text(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """

    def __init__(self, params):
        super(Text, self).__init__(
            dist.Normal,  # prior
            dist.OneHotCategorical,  # likelihood
            dist.Normal,  # posterior
            EncoderText(params),
            DecoderText(params),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'text'
        self.dataSize = torch.Size([TextDataset.size, len(alphabet)])
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def generate(self, runPath, epoch):
        raise NotImplementedError()

    def reconstruct(self, data, runPath, epoch):
        raise NotImplementedError()

    def analyse(self, data, runPath, epoch):
        raise NotImplementedError()
