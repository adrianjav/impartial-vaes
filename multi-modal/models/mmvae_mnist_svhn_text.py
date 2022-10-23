from typing import Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torchvision.utils import save_image

from vis import plot_embeddings, plot_kls_df
from .multimodal import MMVAE
from .vae_mnist import MNIST
from .vae_svhn import SVHN
from .vae_text import Text

from report.helper import Latent_Classifier, SVHN_Classifier, MNIST_Classifier, Text_Classifier


class MNIST_SVHN_TEXT(MMVAE):

    latent_classifiers: nn.ModuleList
    modality_classifiers: Tuple[nn.Module, nn.Module, nn.Module]

    def __init__(self, hparams):
        super(MNIST_SVHN_TEXT, self).__init__(dist.Normal, (MNIST, SVHN, Text), hparams)
        hparams = self.hparams
        grad = {'requires_grad': hparams.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hparams.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, hparams.latent_dim), **grad)  # logvar
        ])

        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if hparams.llik_scaling == 0 else hparams.llik_scaling
        self.vaes[2].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[2].dataSize) \
            if hparams.llik_scaling == 0 else hparams.llik_scaling

        self.modelName = 'mnist-svhn-text'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    def on_test_start(self) -> None:
        @torch.enable_grad()
        def train_latent_classifiers():
            epochs = 30
            device = self._pz_params[0].device

            classifiers = [
                Latent_Classifier(self.hparams.latent_dim, 10).to(device) for _ in range(2**len(self.vaes) - 1)
            ]

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam([{'params': m.parameters()} for m in classifiers], lr=0.001)

            for epoch in range(epochs):  # loop over the dataset multiple times
                for data in self.trainer.datamodule.train_dataloader():
                    xs, labels = data[:-1], data[-1]
                    xs, labels = [x.to(device) for x in xs], labels.to(device)

                    with torch.no_grad():
                        _, zss = self._encode_step(xs, samples=1, all_subsets=True)

                    optimizer.zero_grad()
                    outputs = [classifier(zs.squeeze(0)) for classifier, zs in zip(classifiers, zss)]
                    loss = sum([criterion(output, labels) for output in outputs])
                    loss.backward()
                    optimizer.step()

            return classifiers

        self.latent_classifiers = train_latent_classifiers()
        print('Finished training latent space classifiers.')

        device = self._pz_params[0].device
        mnist_net, svhn_net = MNIST_Classifier().to(device), SVHN_Classifier().to(device)
        text_net = Text_Classifier(self.hparams).to(device)
        mnist_net.load_state_dict(torch.load(self.hparams.experiment_path + '/data/mnist_model.pt'))
        svhn_net.load_state_dict(torch.load(self.hparams.experiment_path + '/data/svhn_model.pt'))
        text_net.load_state_dict(torch.load(self.hparams.experiment_path + '/data/text_model.pt'))
        mnist_net.eval()
        svhn_net.eval()
        text_net.eval()

        self.modality_classifiers = (mnist_net, svhn_net, text_net)
        print('Classifiers loaded.')

    def test_step(self, batch, batch_idx):
        x, labels = batch[:-1], batch[-1]
        samples_post = 1  # TODO
        qz_xs_params, zs = self._encode_step(x, samples=samples_post, all_subsets=True)

        metrics = self._test_step(x, qz_xs_params, zs)
        metrics['latent_classification'] = self._latent_classifcation(zs, labels)
        metrics['cross_coherence'] = self._cross_coherence(zs, labels)
        metrics['joint_coherence'] = self._joint_coherence(x[0].size(0))
        return metrics

    def _latent_classifcation(self, zss, labels):
        labels = labels.expand([zss[0].size(0), -1])
        results = zss[0].new_empty((len(zss), len(self.latent_classifiers)))  # Z x D

        for i, zs in enumerate(zss):
            for j, classifier in enumerate(self.latent_classifiers):
                output = classifier(zs)
                _, predicted = torch.max(output, dim=-1)
                accuracy = (predicted == labels).float()
                results[i, j] = accuracy.mean(dim=0).mean(dim=0)

        return results

    def _cross_coherence(self, zss, labels):
        labels = labels.expand([zss[0].size(0), -1])
        results = zss[0].new_empty((len(self.vaes), len(zss)))  # D x Z

        for d, vae in enumerate(self.vaes):
            for i, zs in enumerate(zss):
                xs = vae.px_z(*vae.dec(zs)).mean

                output = self.modality_classifiers[d](xs.flatten(end_dim=1))
                output = output.view([zs.size(0), zs.size(1), 10])
                _, predicted = torch.max(output, dim=-1)
                accuracy = (predicted == labels).float()
                results[d, i] = accuracy.mean(dim=0).mean(dim=0)

        return results

    def _joint_coherence(self, batch_size):
        samples_generation = batch_size
        zs = self._generation_step(samples=samples_generation).squeeze(1)

        predictions = None
        accuracy = zs.new_ones(zs.size()[:-1])

        for d, vae in enumerate(self.vaes):
            xs = vae.px_z(*vae.dec(zs)).mean
            output = self.modality_classifiers[d](xs)
            _, predicted = torch.max(output, dim=-1)

            if d == 0:
                predictions = predicted
            else:
                accuracy = accuracy + (predicted == predictions).long()

        return (accuracy == len(self.vaes)).float().mean(dim=0)

    def generate(self, runPath, epoch):
        font = ImageFont.truetype('FreeSerif.ttf', 12)
        N = 64
        samples_list = super(MNIST_SVHN_TEXT, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            # wrangle things so they come out tiled
            if i == 2:
                samples = torch.stack(
                    [text_to_pil(d.unsqueeze(0), self.vaes[1].dataSize, alphabet, font) for d in samples], dim=0)
            samples = samples.view(N, *samples.size()[1:])
            save_image(samples,
                       '{}/gen_samples_{}_{:03d}.png'.format(runPath, i, epoch),
                       nrow=int(sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        font = ImageFont.truetype('FreeSerif.ttf', 12)
        recons_mat = super(MNIST_SVHN_TEXT, self).reconstruct([d[:8] for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                _data = data[r][:8].cpu()
                recon = recon.squeeze(0).cpu()
                # resize mnist to 32 and colour. 0 => mnist, 1 => svhn
                _data = _data if r != 0 else resize_img(_data, self.vaes[1].dataSize)
                recon = recon if o != 0 else resize_img(recon, self.vaes[1].dataSize)

                _data = _data if r != 2 else torch.stack([text_to_pil(d.unsqueeze(0), self.vaes[1].dataSize, alphabet, font) for d in _data], dim=0)
                recon = recon if o != 2 else torch.stack([text_to_pil(r.unsqueeze(0), self.vaes[1].dataSize, alphabet, font) for r in recon], dim=0)

                comp = torch.cat([_data, recon])
                save_image(comp, '{}/recon_{}x{}_{:03d}.png'.format(runPath, r, o, epoch))

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(MNIST_SVHN_TEXT, self).analyse(data, K=10)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))


def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)


from datamodules.unimodal import alphabet
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import textwrap


def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded


def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples


def text_to_pil(t, imgsize, alphabet, font, w=32, h=32, linewidth=8):
    blank_img = torch.ones([imgsize[0], w, h])
    pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    text_sample = tensor_to_text(alphabet, t)[0]
    text_sample = ''.join(text_sample).translate({ord('*'): None})
    lines = textwrap.wrap(''.join(text_sample), width=linewidth)
    y_text = h
    num_lines = len(lines)
    for l, line in enumerate(lines):
        width, height = font.getsize(line)
        draw.text((0, (h/2) - (num_lines/2 - l)*height), line, (0, 0, 0), font=font)
        y_text += height
    if imgsize[0] == 3:
        text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                        Image.ANTIALIAS))
    else:
        text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                        Image.ANTIALIAS).convert('L'))
    return text_pil
