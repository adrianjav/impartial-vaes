from functools import reduce
from itertools import chain, combinations
from typing import Sequence

import torch
import torch.nn as nn

from utils import get_mean, kl_divergence
from vis import embed_umap, tensors_to_df

import pytorch_lightning as pl

import torchmoo as moo
from utils import log_mean_exp


def powerset(iterable):
    """ powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
    s = list(iterable)
    sets = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]
    return [set(x) for x in sets]


def poe(params, add_prior=False):  # Modalities, parameters, others
    if add_prior:
        prior = torch.ones_like(params)[:1]
        prior[0, 0].zero_()
        params = torch.cat((params, prior))

    if params.size(0) == 1:
        return params[0]

    mu, std = params.unbind(1)
    var = std.pow(2)
    precision = 1. / var  # precision of i-th Gaussian expert at point x

    pd_mu = torch.sum(mu * precision, dim=0) / torch.sum(precision, dim=0)
    pd_std = torch.sqrt(1. / torch.sum(precision, dim=0))
    return torch.stack((pd_mu, pd_std), dim=0)


class MMVAE(pl.LightningModule):  # TODO rename
    def __init__(self, prior_dist, vaes, hparams):
        super(MMVAE, self).__init__()
        self.save_hyperparameters(hparams)
        hparams = self.hparams
        self.pz = prior_dist
        self.mixture_subsets = {
            'mvae': [set(x for x in range(len(vaes)))],
            'mmvae': [{x} for x in range(len(vaes))],
            'mopoe': powerset(range(len(vaes)))
        }[self.hparams.model]
        self.add_prior = self.hparams.model == 'mvae'

        print('Modalities', {i: v for i, v in enumerate(self.mixture_subsets)})

        assert hparams.looser or not hparams.sample
        self.stl = True
        self.K = hparams.K
        self.idim = {  # M x K x B
            False: {'elbo': (), 'iwae': (1,)},
            True: {'elbo': (0,), 'iwae': (0, 1)}
        }[not hparams.looser][hparams.obj]

        self.vaes = nn.ModuleList([vae(hparams) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self._pz_params = None  # defined in subclass

        num_modalities = len(self.mixture_subsets)

        # MOO
        if hasattr(hparams, 'methods'):

            self.mtl_methods_1 = nn.ModuleList([
                moo.Identity() if hparams.disable_for_loops or 'encoder' not in hparams.mtl_on else self.setup_moo(hparams, len(vaes))
                for _ in range(num_modalities)
            ])
            self.moo_blocks_1 = [moo.MOOForLoop(len(self.vaes), moo_method=method) for method in self.mtl_methods_1]

            self.mtl_methods_2 = nn.ModuleList([
                moo.Identity() if hparams.disable_for_loops or hparams.disable_q or 'encoder' not in hparams.mtl_on else self.setup_moo(hparams, num_modalities)
                for _ in range(num_modalities)
            ])
            self.moo_blocks_2 = [moo.MOOForLoop(num_modalities, moo_method=method) for method in
                                 self.mtl_methods_2]

            self.mtl_methods_3 = nn.ModuleList([  # If I sample there is only one z
                moo.Identity() if hparams.sample or 'decoder' not in hparams.mtl_on else self.setup_moo(hparams, num_modalities)
                for _ in range(len(vaes))
            ])

        for i in range(len(self.vaes)):
            self.vaes[i].dec = moo.MOOModule(self.vaes[i].dec, num_modalities, self.mtl_methods_3[i])

    def setup_moo(self, hparams, num_tasks) -> nn.Module:
        if hparams.methods is None:
            return moo.Identity()

        modules = []
        for method in hparams.methods:
            if method == 'pcgrad':
                modules.append(moo.PCGrad())
            elif method == 'gradvac':
                modules.append(moo.GradVac(hparams.alpha))
            elif method == 'nsgd':
                modules.append(moo.NSGD(num_tasks, hparams.update_at))
            elif method == 'mgda':
                modules.append(moo.MGDAUB())
            elif method == 'cagrad':
                modules.append(moo.CAGrad(hparams.alpha))
            elif method == 'imtl-g':
                modules.append(moo.IMTLG())
            elif method == 'graddrop':
                modules.append(moo.GradDrop())
            elif method == 'gradnorm':
                modules.append(moo.GradNormModified(num_tasks, hparams.alpha, hparams.update_at))
            else:
                raise KeyError(f'Method {method} does not exist.')

        return moo.Compose(*modules) if len(modules) != 0 else moo.Identity()

    def mtl_parameters(self):
        for m in self.mtl_methods_2:
            for p in m.parameters():
                yield p

        for m in self.mtl_methods_1:
            for p in m.parameters():
                yield p

        for m in self.mtl_methods_3:
            for p in m.parameters():
                yield p

    @property
    def pz_params(self):
        return self._pz_params

    def qz_x(self, *args, **kwargs):
        return self.vaes[0].qz_x(*args, **kwargs)

    def forward(self, x, K=1, evaluate=True):  # TODO rework
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        if evaluate:
            for e, zs in enumerate(zss):
                for d, vae in enumerate(self.vaes):
                    if e != d:  # fill-in off-diagonal
                        px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def _encode_step(self, x, *args, samples=None, all_subsets=False):
        samples = samples or self.K
        subsets = powerset(range(len(self.vaes))) if all_subsets else self.mixture_subsets

        qz_xs_params = []
        for i, vae in enumerate(self.vaes):
            qz_x = vae.enc(x[i])
            qz_xs_params.append(qz_x)

        qz_xs_params_subsets, zs = [], []
        for subset in subsets:
            if any(subset.issubset(x) for x in self.mixture_subsets):
                params = poe(torch.stack([qz_xs_params[i] for i in subset], dim=0), add_prior=self.add_prior)
                qz_xs = self.qz_x(*params)
                zs_subset = qz_xs.rsample([samples])
            else:
                params = None
                zs_all = torch.stack([zs[i] for i in subset], dim=0)
                choices = torch.randint(zs_all.size(0), size=(zs_all.size(1), zs_all.size(2))).flatten()
                indexes = zs_all.size(1) * zs_all.size(2) * choices + torch.arange(zs_all.size(1) * zs_all.size(2))
                zs_subset = zs_all.flatten(end_dim=2)[indexes].view(zs[0].size())

            qz_xs_params_subsets.append(params)
            zs.append(zs_subset)

        return qz_xs_params_subsets, zs

    def _multi_modal_loss(self, x, zss, qz_xs_params, *args, stl, idim):
        if stl:  # Stick the landing estimator
            qz_xs = [
                self.qz_x(*[p.detach() for p in qz_x_params]) for qz_x_params in qz_xs_params
            ]
        else:
            qz_xs = [self.qz_x(*qz_x_params) for qz_x_params in qz_xs_params]

        lws, lls = [], []
        for r in range(len(zss)):
            lpz = self.pz(*self.pz_params).log_prob(zss[r]).sum(-1)

            lqz_x = log_mean_exp(torch.stack([
                qz_x.log_prob(zss_r).sum(-1) for qz_x, zss_r in zip(qz_xs, self.moo_blocks_2[r](zss[r]))
            ]))

            lpx_z = []
            for vae in self.vaes:
                vae.dec.moo()

            for d, zs in enumerate(self.moo_blocks_1[r](zss[r])):
                px_z = self.vaes[d].px_z(*self.vaes[d].dec(zs))
                lpx_z.append(
                    px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).mul(self.vaes[d].llik_scaling).sum(-1)
                )
            lpx_z = torch.stack(lpx_z).sum(dim=0)
            lw = lpz + lpx_z - log_mean_exp(lqz_x)

            lws.append(lw)  # K x B
            lls.append(lpx_z)  # K x B
        lw = torch.stack(lws)  # Z x K x B   (First dim -> len(zss))
        lls = torch.stack(lls)  # Z x K x B

        denom = reduce(int.__mul__, [lw.size()[i] for i in range(2) if i not in idim], 1)
        with torch.no_grad():
            if len(idim) > 0:
                grad_wt = (lw - torch.logsumexp(lw, dim=idim, keepdim=True)).exp()  # Z x K x B
            else:
                grad_wt = lw.new_ones(lw.size())

        return -(grad_wt * lw).sum() / denom, lls

    def _run_step(self, x):
        qz_xs, zs = self._encode_step(x)
        loss, lls = self._multi_modal_loss(x, zs, qz_xs, stl=self.stl, idim=self.idim)

        logs = {'loss': loss.item(), 'nll': -lls.mean()}
        return loss, logs

    def training_step(self, batch, batch_idx):
        x = batch[:-1]
        loss, logs = self._run_step(x)
        self.log_dict({f'training/{k}': v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:-1]
        loss, logs = self._run_step(x)
        self.log_dict({f'validation/{k}': v for k, v in logs.items()})
        return loss

    def on_after_backward(self) -> None:
        self.apply(lambda m: getattr(m, 'grad_step', lambda: None)())

    def configure_optimizers(self):
        return torch.optim.Adam(
            [{'params': vae.enc.parameters()} for vae in self.vaes] +
            [{'params': vae.dec.parameters()} for vae in self.vaes] +
            [{'params': self.mtl_parameters(), 'lr': self.hparams.mtl_learning_rate}],
            lr=self.hparams.learning_rate, amsgrad=True
        )

    # Evaluation metrics

    def _generation_step(self, *args, samples):
        pz = self.pz(*self.pz_params)
        return pz.sample([samples])

    def test_step(self, batch, batch_idx):
        x, labels = batch[:-1], batch[-1]
        samples_post = 10  # TODO
        qz_xs_params, zs = self._encode_step(x, samples=samples_post, all_subsets=True)
        return self._test_step(x, qz_xs_params, zs)

    def log_likelihoods(self, batch, mc_samples=10):
        x, labels = batch[:-1], batch[-1]
        _, zs = self._encode_step(x, samples=mc_samples)
        return self.likelihood_per_modality(x, zs)

    def _test_step(self, x, qz_xs_params, zs):
        likelihoods = self.likelihood_per_modality(x, zs)
        kl_matrix_zs = self.kl_matrix_zs(qz_xs_params)
        skl_matrix_zs = kl_matrix_zs + kl_matrix_zs.T
        kl_matrices_xs = self.kl_matrix_xs(x, zs)
        skl_matrices_xs = kl_matrices_xs + torch.transpose(kl_matrices_xs, 1, 2)

        metrics = {
            'likelihoods': likelihoods, 'kl_matrix_zs': kl_matrix_zs, 'skl_matrix_zs': skl_matrix_zs,
            'kl_matrices_xs': kl_matrices_xs, 'skl_matrices_xs': skl_matrices_xs
        }

        samples_generation = 10
        extended_shape = [-1, x[0].size(0)] + [-1 for _ in range(zs[0].ndim)[2:]]
        zs = self._generation_step(samples=samples_generation)
        zs = zs.expand(extended_shape).contiguous()
        zs = [zs]

        likelihoods = self.likelihood_per_modality(x, zs)
        kl_matrices_xs = self.kl_matrix_xs(x, zs)
        skl_matrices_xs = kl_matrices_xs + torch.transpose(kl_matrices_xs, 1, 2)

        metrics.update({
            'likelihoods_joint': likelihoods, 'kl_matrices_xs_joint': kl_matrices_xs,
            'skl_matrices_xs_joint': skl_matrices_xs
        })

        return metrics

    def test_epoch_end(self, outputs: dict) -> None:
        keys = outputs[0].keys()
        metrics = {k: torch.stack([out[k] for out in outputs][:-1]).mean(dim=0) for k in keys}  # TODO -1

        def linearize(name, tensor):
            if tensor.numel() == 1:
                return {name: tensor.item()}

            if len(tensor) == 1:
                return linearize(name, tensor[0])

            result = {}
            for i, t in enumerate(tensor):
                result.update(linearize(f'{name}_{i}', t))

            return result

        result = {}
        for k, v in metrics.items():
            result.update(linearize(k, v))

        self.log_dict(result)

    def likelihood_per_modality(self, x, zs):
        likelihoods = []

        for z in zs:
            lpxs_z = []
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(z))
                lpx_z = px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).sum(-1)  # Sum over data dimensions
                lpx_z = lpx_z.mean(dim=0).mean(dim=0)  # Average over z samples and batch samples

                lpxs_z.append(lpx_z)

            likelihoods.append(torch.stack(lpxs_z, dim=0))
        return torch.stack(likelihoods, dim=0)  # Z x D

    def kl_matrix_zs(self, qz_xs_params: Sequence[torch.Tensor]):
        Z = sum([x is not None for x in qz_xs_params])
        kl_matrix = qz_xs_params[0].new_zeros((Z, Z))

        x = 0
        for i, params_i in enumerate(qz_xs_params):
            if params_i is not None:
                y = 0
                q_i = self.qz_x(*params_i)
                for j, params_j in enumerate(qz_xs_params):
                    if params_j is not None:
                        if x != y:
                            q_j = self.qz_x(*params_j)
                            kl_matrix[x, y] = kl_divergence(q_i, q_j).sum(dim=1).mean(dim=0)
                        y += 1
                x += 1

        return kl_matrix

    def kl_matrix_xs(self, x, zs):
        D, Z = len(x), len(zs)
        pxs_z = []  # D x Z

        for vae in self.vaes:
            pxs_z.append([])
            for z in zs:
                px_z = vae.px_z(*vae.dec(z))
                pxs_z[-1].append(px_z)

        kls = []
        for d, pxd_z in enumerate(pxs_z):
            kls.append(x[0].new_zeros((Z, Z)))
            for i1 in range(Z):
                for i2 in range(Z):
                    if i1 != i2:
                        kl = kl_divergence(pxd_z[i1], pxd_z[i2]).flatten(start_dim=2).sum(-1)
                        kl = kl.mean(dim=0).mean(dim=0)  # mean over zs samples and batch samples
                        kls[-1][i1, i2] = kl

        return torch.stack(kls, dim=0)

    # TODO here

    def generate(self, N):
        self.eval()
        with torch.no_grad():
            data = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(latents))
                data.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons

    def analyse(self, data, K):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data, K=K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df
