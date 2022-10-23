from functools import reduce

import torch
import torch.nn.functional as F
import torch.distributions.utils
from numpy import prod

from utils import log_mean_exp, is_multidata, kl_divergence


class BreakPoint:
    __slots__ = ['sink', 'source']

    def __init__(self, tensor):
        self.sink = tensor
        self.source = tensor.detach().clone()
        self.source.requires_grad = True

    def fire(self):
        self.sink.backward(self.source.grad, retain_graph=True)



# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def elbo(model, x, K=1):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()


def _iwae(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1) - lqz_x


def iwae(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _dreg(model, x, K):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.llik_scaling
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs


def dreg(model, x, K, regs=None):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw, zs = zip(*[_dreg(model, _x, K) for _x in x.split(S)])
    lw = torch.cat(lw, 1)  # concat on batch
    zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()


# multi-modal variants


def m_elbo(model, x, K=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * model.vaes[d].llik_scaling).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            lpx_zs.append(lwt.exp() * lpx_z)
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()


def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


def multi_modal_loss(model, x, zss, qz_xs, stl, dim):
    if stl:  # Stick the landing estimator (biased)
        qz_xs = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]

    lws = []
    for r in range(len(zss)):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)

        lqz_x = log_mean_exp(torch.stack([
            qz_x.log_prob(zss_r).sum(-1) for qz_x, zss_r in zip(qz_xs, model.moo_blocks[r](zss[r]))
        ]))

        lpx_z = []
        for vae in model.vaes:
            vae.dec.moo()

        for d, zs in enumerate(model.moo_blocks2[r](zss[r])):
            px_z = model.vaes[d].px_z(*model.vaes[d].dec(zs))
            lpx_z.append(
                px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).mul(model.vaes[d].llik_scaling).sum(-1)
            )
        lpx_z = torch.stack(lpx_z).sum(dim=0)
        lw = lpz + lpx_z - log_mean_exp(lqz_x)

        lws.append(lw)  # K x B
    lw = torch.stack(lws)  # Z x K x B   (First dim -> len(zss))

    denom = reduce(int.__mul__, [lw.size()[i] for i in range(2) if i not in dim], 1)
    with torch.no_grad():
        if len(dim) > 0:
            grad_wt = (lw - torch.logsumexp(lw, dim=dim, keepdim=True)).exp()  # Z x K x B
        else:
            grad_wt = lw.new_ones(lw.size())

    return -(grad_wt * lw).sum() / denom


def multi_modal_loss_splitted(model, x, zss, qz_xs, stl, dim):
    if stl:  # Stick the landing estimator (biased)
        qz_xs = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]

    breakpoints = []

    lws, lqz_xs, lpx_zs, lpzs = [], [], [], []
    for r in range(len(zss)):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)

        breakpoints.append(BreakPoint(model.moo_blocks[r](zss[r])))
        lqz_x = []
        for qz_x, zs in zip(qz_xs, breakpoints[-1].source):
            lqz_x.append(qz_x.log_prob(zs).sum(-1))
        lqz_x = torch.stack(lqz_x)

        breakpoints.append(BreakPoint(model.moo_blocks2[r](zss[r])))
        lpx_z = []
        for d, zs in enumerate(breakpoints[-1].source):
            px_z = model.vaes[d].px_z(*model.vaes[d].dec(zs))
            lpx_z.append(
                px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).mul(model.vaes[d].llik_scaling).sum(-1)
            )
        lpx_z = torch.stack(lpx_z)
        lw = lpz + lpx_z.sum(0) - log_mean_exp(lqz_x)

        lws.append(lw)  # K x B
        lqz_xs.append(lqz_x)  # M x K x B
        lpx_zs.append(lpx_z)  # M x K x B
        lpzs.append(lpz)

    lw = torch.stack(lws)  # Z x K x B
    lqz_xs = torch.stack(lqz_xs)  # Z x M x K x B   (First dim -> len(zss))
    lpx_zs = torch.stack(lpx_zs)  # Z x M x K x B
    lpzs = torch.stack(lpzs)  # Z x K x B

    # Z, K, B = lw.size()
    denom = reduce(int.__mul__, [lw.size()[i] for i in range(2) if i not in dim], 1)
    with torch.no_grad():
        if len(dim) > 0:
            grad_wt = (lw - torch.logsumexp(lw, dim=dim, keepdim=True)).exp()  # Z x K x B
        else:
            grad_wt = lw.new_ones(lw.size())

        # This one always add up to the same dimension
        gamma = (lqz_xs - torch.logsumexp(lqz_xs, 1, keepdim=True)).exp()  # Z x M x K x B

    loss_regularizer = -(grad_wt * lpzs).sum() / denom

    losses = [[] for _ in range(len(model.vaes))]
    for r in range(len(model.vaes)):
        for d in range(len(zss)):
            loss_r_d = grad_wt[d] * (lpx_zs[d][r] - gamma[d][r] * lqz_xs[d][r])  # K x B
            loss_r_d = -loss_r_d.sum() / denom
            losses[r].append(loss_r_d)

    return loss_regularizer, losses, breakpoints


# Stratified


def m_elbo_naive_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=False, dim=(0,))


def m_elbo_naive_looser_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=False, dim=())


def m_elbo_naive_stl_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=True, dim=(0,))


def m_elbo_naive_stl_looser_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=True, dim=())


def m_iwae_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=False, dim=(0, 1))


def m_iwae_looser_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=False, dim=(1,))


def m_iwae_stl_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=True, dim=(0, 1))


def m_iwae_stl_looser_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    return multi_modal_loss(model, x, zss, qz_xs, stl=True, dim=(1,))


# Sampling


def sample_mixture(zss, hard=True):
    logits = torch.distributions.utils.probs_to_logits(zss[0].new_ones((zss[0].size(1), len(zss),)))
    mixing = F.gumbel_softmax(logits, 1., hard=hard)
    return sum([zs * m.unsqueeze(-1) for zs, m in zip(zss, mixing.unbind(dim=-1))]).unsqueeze(0)


def m_elbo_naive_sample_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    zss = sample_mixture(zss, hard=True)
    return multi_modal_loss(model, x, zss, qz_xs, stl=False, dim=(0,))


def m_elbo_naive_stl_sample_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    zss = sample_mixture(zss, hard=True)
    return multi_modal_loss(model, x, zss, qz_xs, stl=True, dim=(0,))


def m_iwae_sample_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    zss = sample_mixture(zss, hard=True)
    return multi_modal_loss(model, x, zss, qz_xs, stl=False, dim=(0, 1))


def m_iwae_stl_sample_mtl(model, x, K=1):
    qz_xs, _, zss = model(x, K, evaluate=False)  # M x K x B
    zss = sample_mixture(zss, hard=True)
    return multi_modal_loss(model, x, zss, qz_xs, stl=True, dim=(0, 1))


