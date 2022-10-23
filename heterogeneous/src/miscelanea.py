import time
import functools

import torch
import torch.distributions as dists

import src.distributions as my_dists


def fix_seed(seed) -> None:
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()

    return x_one_hot


def get_distribution_by_name(name):
    return {'normal': dists.Normal, 'gamma': dists.Gamma, 'bernoulli': dists.Bernoulli,
            'categorical': dists.Categorical, 'lognormal': dists.LogNormal,
            'poisson': dists.Poisson, 'exponential': dists.Exponential}[name]


def print_epoch_value(engine, metrics, trainer, max_epochs, fn=lambda x: x):
    if (trainer.state.epoch - 1) % (max(1, max_epochs // 40)) != 0:
        return

    msg = f'Epoch {trainer.state.epoch} of {max_epochs}:'

    for name in metrics:
        value = fn(engine.state.metrics[name])
        msg += ' {} {:.5f}' if isinstance(value, float) else ' {} {}'

        if isinstance(value, torch.Tensor):
            value = value.tolist()

        msg = msg.format(name, value)

    print(msg)


def calculate_mie(engine, model, prob_model, dataset, device):
    model.eval()
    epoch = engine.state.epoch
    mean = lambda x: sum(x).item() / len(x)

    if epoch % (engine.state.max_epochs / 4) != 0:
        return

    data = dataset[:][0]
    observed_mask = dataset[:][1]
    nan_mask = dataset[:][2]
    missing_mask = ((1 - observed_mask.long()) + nan_mask.long()) == 2

    mask_bc = dataset[:][1].to(device)
    pred = model([dataset[:][0].to(device), mask_bc, None]).cpu()

    observed_error = imputation_error(prob_model, pred, data, observed_mask)

    nominal_error = [e for e, [_, d] in zip(observed_error, prob_model.gathered) if d.real_dist.is_discrete]
    nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
    numerical_error = [e for e, [_, d] in zip(observed_error, prob_model.gathered) if d.real_dist.is_continuous]
    numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

    print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] observed imputation error:')
    for i, error in enumerate(observed_error):
        print(f'[dim={i}] {error}')
    print('nominal  :', nominal_error)
    print('numerical:', numerical_error)
    print('total    :', mean(observed_error))
    print('')

    if missing_mask.any():
        missing_error = imputation_error(prob_model, pred, data, missing_mask)

        nominal_error = [e for e, [_, d] in zip(missing_error, prob_model.gathered) if d.real_dist.is_discrete]
        nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
        numerical_error = [e for e, [_, d] in zip(missing_error, prob_model.gathered) if d.real_dist.is_continuous]
        numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

        print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] missing imputation error:')
        for i, error in enumerate(missing_error):
            print(f'[dim={i}] {error}')
        print('nominal  : ', nominal_error)
        print('numerical: ', numerical_error)
        print('total: ', mean(missing_error))
        print('')


def nrmse(pred, target, mask):  # for numerical variables
    norm_term = torch.max(target) - torch.min(target)
    new_pred = torch.masked_select(pred, mask.bool())
    new_target = torch.masked_select(target, mask.bool())

    return torch.sqrt(torch.nn.functional.mse_loss(new_pred, new_target)) / norm_term


def accuracy(pred, target, mask):  # for categorical variables
    return torch.sum((pred != target).float() * mask) / mask.sum()


def displacement(pred, target, mask, size):  # for ordinal variables
    diff = (target - pred).abs() * mask / size
    return diff.sum() / mask.sum()


def imputation_error(prob_model, pred, target, mask):
    mask = mask.float()

    errors = []
    for i, [_, dist] in enumerate(prob_model.gathered):
        pos = prob_model.gathered_index(i)

        if isinstance(dist.real_dist, my_dists.Categorical) or isinstance(dist.real_dist, my_dists.Bernoulli):
            errors.append(accuracy(pred[:, i], target[:, i], mask[:, pos]))
        else:  # numerical
            errors.append(nrmse(pred[:, i], target[:, i], mask[:, pos]))

    return errors


@torch.no_grad()
def calculate_ll(engine, model, prob_model, dataset, device):
    model.eval()
    epoch = engine.state.epoch
    mean = lambda x: sum(x).item() / len(x)

    if epoch % (engine.state.max_epochs / 4) != 0:
        return

    # data = dataset.original_data
    observed_mask = dataset[:][1]
    nan_mask = dataset[:][2]
    missing_mask = ((1 - observed_mask.long()) + nan_mask.long()) == 2

    observed_mask_bc = dataset[:][1]

    log_prob = model.log_likelihood_real(dataset[:][0].to(device), observed_mask_bc.to(device))
    log_prob = log_prob.cpu()

    observed_log_prob = (log_prob * observed_mask).sum(dim=0)
    for i in range(len(prob_model.gathered)):
        observed_log_prob[i] = observed_log_prob[i] / observed_mask[:, i].sum()

    nominal_error = [e for e, [_, d] in zip(observed_log_prob, prob_model.gathered) if d.real_dist.is_discrete]
    nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
    numerical_error = [e for e, [_, d] in zip(observed_log_prob, prob_model.gathered) if d.real_dist.is_continuous]
    numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

    print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] observed log-likelihood:')
    for i, error in enumerate(observed_log_prob):
        print(f'[dim={i}] {error}')
    print('nominal  :', nominal_error)
    print('numerical:', numerical_error)
    print('total    :', mean(observed_log_prob))
    print('')

    if missing_mask.any():
        missing_log_prob = (log_prob * missing_mask).sum(dim=0)
        for i in range(len(prob_model.gathered)):
            missing_log_prob[i] = missing_log_prob[i] / max(missing_mask[:, i].sum(), 1)

        nominal_error = [e for e, [_, d] in zip(missing_log_prob, prob_model.gathered) if d.real_dist.is_discrete]
        nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
        numerical_error = [e for e, [_, d] in zip(missing_log_prob, prob_model.gathered) if d.real_dist.is_continuous]
        numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

        print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] missing log-likelihood:')
        for i, error in enumerate(missing_log_prob):
            print(f'[dim={i}] {error}')
        print('nominal  :', nominal_error)
        print('numerical:', numerical_error)
        print('total    :', mean(missing_log_prob))
        print('')


def timed(func):
    @functools.wraps(func)
    def timed_(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f'{func.__name__} completed in {end - start :.3f} seconds.')
        print()

        return result
    return timed_


@torch.no_grad()
def test_mie_ll(model, prob_model, dataset, device, title='Test', missing=False):
    model.eval()
    mean = lambda x: sum(x).item() / len(x)

    data = dataset[:][0]
    mask = dataset[:][2] if not missing else dataset[:][1]

    mask_bc = dataset[:][2] if not missing else dataset[:][1]

    pred = model([dataset[:][0].to(device), mask_bc, None], mode='reconstruct_mode').cpu()

    log_prob = model.log_likelihood_real(dataset[:][0].to(device), mask_bc.to(device))

    if missing:
        mask = ((1 - mask.long()) + dataset[:][2].long()) == 2
        assert mask.sum() > 0

    log_prob = (log_prob * mask).sum(dim=0).cpu() / mask.sum(dim=0)
    error = imputation_error(prob_model, pred, data, mask)

    nominal_ll = [e for e, [_, d] in zip(log_prob, prob_model.gathered) if d.real_dist.is_discrete]
    nominal_ll = mean(nominal_ll) if len(nominal_ll) > 0 else 0.
    numerical_ll = [e for e, [_, d] in zip(log_prob, prob_model.gathered) if d.real_dist.is_continuous]
    numerical_ll = mean(numerical_ll) if len(numerical_ll) > 0 else 0.

    print(f'[{title}] log-likelihood:')
    for i, ll in enumerate(log_prob):
        print(f'[dim={i}] {ll}')
    print('nominal  :', nominal_ll)
    print('numerical:', numerical_ll)
    print('total    :', mean(log_prob))
    print('')

    nominal_error = [e for e, [_, d] in zip(error, prob_model.gathered) if d.real_dist.is_discrete]
    nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
    numerical_error = [e for e, [_, d] in zip(error, prob_model.gathered) if d.real_dist.is_continuous]
    numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

    print(f'[{title}] error:')
    for i, e in enumerate(error):
        print(f'[dim={i}] {e}')
    print('nominal  :', nominal_error)
    print('numerical:', numerical_error)
    print('total    :', mean(error))
    print('')
