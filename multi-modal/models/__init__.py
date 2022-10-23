from .mmvae_mnist_svhn_text import MNIST_SVHN_TEXT as VAE_mnist_svhn_text
from .vae_mnist import MNIST as VAE_mnist
from .vae_svhn import SVHN as VAE_svhn

__all__ = [VAE_mnist_svhn_text, VAE_mnist, VAE_svhn]
