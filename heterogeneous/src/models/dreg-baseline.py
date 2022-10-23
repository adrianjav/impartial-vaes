import torch

from .vae import VAE


class DREG(VAE):
    def __init__(self, prob_model, hparams):
        super().__init__(prob_model, hparams)
        self.n_samples = hparams.samples

    def _run_step(self, x, mask):
        z_params = self.encoder(x if mask is None else x * mask.float())
        z = self.encoder.q_z(*z_params).rsample([self.n_samples])

        y = self.decoder_shared(z)
        x_params = [head(y_i) for head, y_i in zip(self.heads, self.moo_block(y))]

        x_scaled = self.prob_model >> x
        x_scaled = x_scaled.unsqueeze(dim=0).tile((self.n_samples, 1, 1))
        mask = mask.unsqueeze(dim=0).tile((self.n_samples, 1, 1))

        # samples x batch_size x D
        log_px_z = [self.log_likelihood(x_scaled, mask, i, params_i) for i, params_i in enumerate(x_params)]

        log_pz = self.prior_z.log_prob(z).sum(dim=-1)  # samples x batch_size
        z_params = [param_i.detach() for param_i in z_params]
        log_qz_x = self.encoder.q_z(*z_params).log_prob(z).sum(dim=-1)  # samples x batch_size
        kl_z = log_qz_x - log_pz

        return log_px_z, kl_z, z

    def _step(self, batch, batch_idx):
        x, mask, _ = batch
        log_px_z, kl_z, z = self._run_step(x, mask)

        lw = sum(log_px_z) - kl_z  # samples x batch_size
        with torch.no_grad():
            w_tilde = torch.exp(lw - torch.logsumexp(lw, dim=0, keepdim=True))

        if z.requires_grad:
            z.register_hook(lambda g: w_tilde.unsqueeze(-1) * g)

        loss = torch.sum(w_tilde * lw, dim=0)
        loss = -loss.sum(dim=0)
        assert loss.size() == torch.Size([])

        logs = dict()
        logs['loss'] = loss / x.size(0)

        with torch.no_grad():
            log_prob = (self.log_likelihood_real(x, mask) * mask).sum(dim=0) / mask.sum(dim=0)
            logs['re'] = -log_prob.mean(dim=0)
            logs['kl'] = kl_z.mean(dim=0).mean(dim=0)
            logs.update({f'll_{i}': l_i.item() for i, l_i in enumerate(log_prob)})

        return loss, logs
