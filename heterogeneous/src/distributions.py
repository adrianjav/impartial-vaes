from __future__ import annotations

from functools import reduce
from typing import Iterable

import torch
import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs, broadcast_all
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical

from .miscelanea import to_one_hot


class GumbelDistribution(ExpRelaxedCategorical):
    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        return OneHotCategorical(probs=self.probs).sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return torch.exp(super().rsample(sample_shape))

    @property
    def mean(self):
        return self.probs

    def expand(self, batch_shape, _instance=None):
        return super().expand(batch_shape[:-1], _instance)

    def log_prob(self, value):
        return OneHotCategorical(probs=self.probs).log_prob(value)


def get_distribution_by_name(name):
    return {
        'normal': Normal, 'lognormal': LogNormal, 'gamma': Gamma, 'exponential': Exponential,
        'bernoulli': Bernoulli, 'poisson': Poisson, 'categorical': Categorical, 'ordinal': Categorical,
        'geometric': Geometric, 'negative_binomial': NegativeBinomial, 'truncated_normal': TruncatedNormal,
        'bernoullitrick': {
            'categorical': CategoricalBernoulliTrick  # , 'ordinal': OrdinalBernoulliTrick
        },
        'gammatrick' : {
            'bernoulli': BernoulliGammaTrick, 'poisson': PoissonGammaTrick,
            'categorical': CategoricalGammaTrick  # , 'ordinal': OrdinalGammaTrick
        }
        }[name]


class Base(object):
    def __init__(self):
        self._weight = torch.tensor([1.0])
        self.arg_constraints = {}
        self.size = 1

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, torch.Tensor) and isinstance(value, Iterable):
            assert len(value) == 1, value
            value = iter(value)

        self._weight = value if isinstance(value, torch.Tensor) else torch.tensor([value])

    @property
    def expanded_weight(self):
        return reduce(list.__add__, [[w] * len(self[i].f) for i,w in enumerate(self.weight)])

    @property
    def parameters(self):
        return list(self.dist.arg_constraints.keys())

    @property
    def real_parameters(self):
        return self.real_dist.real_parameters if id(self) != id(self.real_dist) else self.parameters

    def __getitem__(self, item):
        assert item == 0
        return self

    def preprocess_data(self, x, mask=None):
        return x,

    def scale_data(self, x, weight=None):
        weight = weight or self.weight
        return x * weight

    def unscale_data(self, x, weight=None):
        weight = weight or self.weight
        return x / weight

    @property
    def f(self):
        raise NotImplementedError()

    def sample(self, size, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).sample(torch.Size([size]))

    def impute(self, etas):
        raise NotImplementedError()
        # real_params = self.to_real_params(etas)
        # real_params = dict(zip(self.real_parameters, real_params))
        # return self.real_dist.dist(**real_params).mean

    def mean(self, etas):
        params = self.to_params(etas)
        params = dict(zip(self.parameters, params))
        return self.dist(**params).mean

    def to_text(self, etas):
        params = self.to_real_params(etas)
        params = [x.cpu().tolist() for x in params]
        params = dict(zip(self.real_parameters, params))
        try:
            mean = self.mean(etas).item()
        except NotImplementedError:
            mean = None

        return f'{self.real_dist} params={params}' + (f' mean={mean}' if mean is not None else '')

    def params_from_data(self, x):
        raise NotImplementedError()

    def real_params_from_data(self, x):
        etas = self.real_dist.params_from_data(x)
        return self.real_dist.to_real_params(etas)

    @property
    def real_dist(self) -> Base:
        return self

    def to_real_params(self, etas):
        return self.to_params(etas)

    @property
    def num_params(self):
        return len(self.arg_constraints)

    @property
    def size_params(self):
        return [1] * self.num_params

    @property
    def num_suff_stats(self):
        return self.num_params

    @property
    def num_dists(self):
        return 1

    def log_prob(self, x, etas):
        params = self.to_params(etas)
        params = dict(zip(self.parameters, params))
        return self.dist(**params).log_prob(x)

    def real_log_prob(self, x, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).log_prob(x)

    @property
    def dist(self):
        raise NotImplementedError()

    def unscale_params(self, etas):
        c = torch.ones_like(etas)
        for i, f in enumerate(self.f):
            c[i].mul_(f(self.expanded_weight[i]).item())
        return etas * c

    def scale_params(self, etas):
        c = torch.ones_like(etas)
        for i, f in enumerate(self.f):
            c[i].mul_(f(self.expanded_weight[i]).item())
        return etas / c

    def __str__(self):
        raise NotImplementedError()

    def to_params(self, etas):
        raise NotImplementedError()

    def to_naturals(self, params):
        raise NotImplementedError()

    @property
    def is_discrete(self):
        raise NotImplementedError()

    @property
    def is_continuous(self):
        return not self.is_discrete

    def __rshift__(self, data):
        return self.scale_data(data)

    def __lshift__(self, etas):
        return self.unscale_params(etas)


class Normal(Base):
    def __init__(self):
        super(Normal, self).__init__()

        self.arg_constraints = [
            constraints.real,  # eta1
            constraints.less_than(0)  # eta2
        ]

    @property
    def is_discrete(self):
        return False

    @property
    def dist(self):
        return dist.Normal

    @property
    def f(self):
        return [lambda w: w, lambda w: w**2]

    def params_from_data(self, x):
        return self.to_naturals([x.mean(), x.std()])

    def to_params(self, etas):
        eta1, eta2 = etas
        return -0.5 * eta1 / eta2, torch.sqrt(-0.5 / eta2)

    def to_naturals(self, params):
        loc, std = params

        eta2 = -0.5 / std ** 2
        eta1 = -2 * loc * eta2

        return eta1, eta2

    def impute(self, etas):
        return self.mean(etas)

    def __str__(self):
        return 'normal'


class TruncatedNormal(Normal):
    @property
    def is_discrete(self):
        return True

    def log_prob(self, x, etas):
        params = self.to_params(etas)
        params = dict(zip(self.parameters, params))
        dist = self.dist(**params)

        log_prob = torch.log(torch.clamp(dist.cdf(x+1) - dist.cdf(x), min=1e-20))  # Slicing
        mask = x <= 0
        log_prob[mask] = dist.cdf(torch.zeros_like(log_prob))[mask].clamp_min(1e-20).log()

        return log_prob

    def real_log_prob(self, x, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        dist = self.real_dist.dist(**real_params)

        log_prob = torch.log(torch.clamp(dist.cdf(x+1) - dist.cdf(x), min=1e-20))  # Slicing
        mask = x <= 0
        log_prob[mask] = dist.cdf(torch.zeros_like(log_prob))[mask].clamp_min(1e-20).log()

        return log_prob

    def impute(self, etas):
        return self.mean(etas).floor().clamp_min(0)

    def __str__(self):
        return 'truncated_normal'


class LogNormal(Normal):
    def scale_data(self, x, weight=None):
        weight = self.weight if weight is None else weight
        return torch.clamp(torch.pow(x, weight), min=1e-20, max=1e20)

    def unscale_data(self, x, weight=None):
        weight = self.weight if weight is None else weight
        return torch.clamp(torch.pow(x, 1./weight), min=1e-20, max=1e20)

    @property
    def dist(self):
        return dist.LogNormal

    def params_from_data(self, x):
        return super().params_from_data(torch.log(x))

    def sample(self, size, etas):
        return torch.clamp(super().sample(size, etas), min=1e-20, max=1e20)

    def impute(self, etas):
        mu, sigma = self.to_real_params(etas)
        return torch.clamp(torch.exp(mu - sigma**2), min=1e-20, max=1e20)

    def __str__(self):
        return 'lognormal'


class Gamma(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            constraints.greater_than(-1),  # eta1
            constraints.less_than(0)  # eta2
        ]

    @property
    def dist(self):
        return dist.Gamma

    @property
    def f(self):
        return [lambda w: torch.ones_like(w), lambda w: w]

    @property
    def is_discrete(self):
        return False

    def params_from_data(self, x):
        mean, meanlog = x.mean(), x.log().mean()
        s = mean.log() - meanlog

        shape = (3 - s + ((s-3)**2 + 24*s).sqrt()) / (12 * s)
        for _ in range(50):
            shape = shape - (shape.log() - torch.digamma(shape) - s) / (1 / shape - torch.polygamma(1, shape))

        concentration = shape
        rate = shape / mean

        eta1 = concentration - 1
        eta2 = -rate

        return eta1, eta2

    def to_params(self, etas):
        eta1, eta2 = etas

        return eta1 + 1, -eta2

    def impute(self, etas):
        alpha, beta = self.to_real_params(etas)
        return torch.clamp((alpha - 1) / beta, min=0.0)

    def __str__(self):
        return 'gamma'


class Exponential(Base):
    def __init__(self):
        super(Exponential, self).__init__()

        self.arg_constraints = [
            constraints.less_than(0)  # eta1
        ]

    @property
    def dist(self):
        return dist.Exponential

    @property
    def is_discrete(self):
        return False

    @property
    def f(self):
        return [lambda w: w]

    def params_from_data(self, x):
        mean = x.mean()
        return -1 / mean,

    def to_params(self, etas):
        return -etas[0],

    def impute(self, etas):
        raise NotImplementedError()

    def __str__(self):
        return "exponential"


class Bernoulli(Base):
    def __init__(self):
        super().__init__()
        self.size = 2
        self.arg_constraints = [
            constraints.real
        ]

    @property
    def dist(self):
        return dist.Bernoulli

    @property
    def is_discrete(self):
        return True

    @property
    def parameters(self):
        return 'logits',

    @property
    def real_parameters(self):
        return 'probs',

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return probs_to_logits(x.mean(), is_binary=True),

    def to_params(self, etas):
        return etas[0],

    def to_real_params(self, etas):
        return logits_to_probs(self.to_params(etas)[0], is_binary=True),

    def impute(self, etas):
        probs = self.to_real_params(etas)[0]

        return (probs >= 0.5).float()

    def __str__(self):
        return 'bernoulli'


class Poisson(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            constraints.real
        ]

    @property
    def dist(self):
        return dist.Poisson

    @property
    def is_discrete(self):
        return True

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return torch.log(torch.clamp(x.mean(), min=1e-20)),

    def to_params(self, etas):
        return torch.exp(etas[0]).clamp(min=1e-6, max=1e20),  # TODO

    def impute(self, etas):
        rate = self.to_real_params(etas)[0]
        return rate.floor()

    def __str__(self):
        return 'poisson'


class Geometric(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            constraints.real
        ]

    @property
    def dist(self):
        return dist.Geometric

    @property
    def is_discrete(self):
        return True

    @property
    def parameters(self):
        return 'logits',

    @property
    def real_parameters(self):
        return 'probs',

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return probs_to_logits(torch.reciprocal(x.mean() + 1), is_binary=True),

    def to_params(self, etas):
        return etas[0],

    def to_real_params(self, etas):
        return logits_to_probs(self.to_params(etas)[0], is_binary=True),

    def impute(self, etas):
        return torch.floor(self.mean(etas))

    def __str__(self):
        return 'geometric'


class NegativeBinomial(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            constraints.greater_than(0),
            constraints.real
        ]

    @property
    def dist(self):
        return dist.NegativeBinomial

    @property
    def is_discrete(self):
        return True

    @property
    def parameters(self):
        return 'total_count', 'logits'

    @property
    def real_parameters(self):
        return 'total_count', 'probs'

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)] * 2

    def params_from_data(self, x):
        m, s = x.mean(), x.std()
        probs = torch.clamp(1. - m / s**2, 0, 1)
        logits = probs_to_logits(probs, is_binary=True)
        total_count = torch.clamp(m * (1 - probs) / probs, 1e-20)
        return total_count, logits

    def to_params(self, etas):
        return etas

    def to_real_params(self, etas):
        return etas[0], logits_to_probs(self.to_params(etas)[1], is_binary=True),

    def impute(self, etas):
        total_count, probs = self.to_real_params(etas)
        mode = torch.floor(probs * (total_count - 1) / (1 - probs))
        mask = total_count <= 1.
        mode = torch.masked_fill(mode, mask, 0.)
        return mode

    def __str__(self):
        return 'negative binomial'


class TruncatedNormalDistribution(dist.Distribution):
    arg_constraints = {'minimum_value': constraints.real,
                       'maximum_value': constraints.real,
                       'loc': constraints.real,
                       'scale': constraints.positive}  # See https://en.wikipedia.org/wiki/Truncated_normal_distribution
    support = constraints.real
    has_rsample = True

    def __init__(self, minimum_value: torch.Tensor, maximum_value: torch.Tensor, loc: torch.Tensor,
                 scale: torch.Tensor, validate_args: bool = True):
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.loc = loc
        self.scale = scale

        self.minimum_value, self.maximum_value, self.loc, self.scale = broadcast_all(
            self.minimum_value, self.maximum_value, self.loc, self.scale
        )
        assert torch.all(self.minimum_value <= self.maximum_value)

        self._normal = dist.Normal(torch.zeros_like(loc), 1.)

        self.alpha = (self.minimum_value - self.loc) / self.scale
        self.beta = (self.maximum_value - self.loc) / self.scale
        self.normalizing_constant = self._normal.cdf(self.beta) - self._normal.cdf(self.alpha)
        self.normalizing_constant = torch.clamp_min(self.normalizing_constant, 1e-20)

        batch_shape = self.loc.size()
        super(TruncatedNormalDistribution, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        extra_loc = (self._normal.log_prob(self.alpha).exp() - self._normal.log_prob(self.beta).exp())
        extra_loc = extra_loc / self.normalizing_constant * self.scale
        return self.loc + extra_loc

    @property
    def param_shape(self):
        return self.loc.size()

    @property
    def variance(self):
        # TODO Not needed
        raise NotImplementedError()

    def cdf(self, value):
        return (self._normal.cdf((value - self.loc) / self.scale) - self._normal.cdf(self.alpha)) \
               / self.normalizing_constant

    def sample(self, sample_shape=torch.Size()):
        samples = dist.Uniform(torch.zeros_like(self.loc), torch.ones_like(self.loc)).sample(sample_shape)
        cdf_alpha, cdf_beta = self._normal.cdf(self.alpha), self._normal.cdf(self.beta)
        samples = self._normal.icdf(cdf_alpha + samples * (cdf_beta - cdf_alpha)) * self.scale + self.loc
        return samples

    def log_prob(self, value):
        assert torch.all(self.minimum_value <= value) and torch.all(value <= self.maximum_value)
        if self._validate_args:
            self._validate_sample(value)

        log_prob = self._normal.log_prob((value - self.loc) / self.scale) - self.scale.log()
        log_prob -= self.normalizing_constant.log()
        return log_prob


class TruncatedNormal2(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            # constraints.real,  # min
            # constraints.real,  # max
            constraints.real,  # loc
            constraints.positive  # scale
        ]

    @property
    def dist(self):
        return TruncatedNormalDistribution

    @property
    def is_discrete(self):
        return True

    @property
    def parameters(self):
        return 'minimum_value', 'maximum_value', 'loc', 'scale'

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)] * 2

    def params_from_data(self, x):
        raise NotImplementedError

    def to_params(self, etas):
        return 0., 1., etas[0], etas[1]

    def impute(self, etas):
        mode = self.mean(etas)

        mask1 = etas[0] < 0.
        mode = torch.masked_fill(mode, mask1, 0.)

        mask2 = etas[0] > 1.
        mode = torch.masked_fill(mode, mask2, 1.)

        # return torch.floor(mode)
        return mode

    def __str__(self):
        return 'truncated_normal'


class BernoulliGammaTrick(Gamma):
    def __init__(self):
        super().__init__()
        self.noise_dist = dist.Beta(1.1, 30)

    @property
    def real_dist(self) -> Base:
        return Bernoulli()

    def preprocess_data(self, x, mask=None):
        x = super(BernoulliGammaTrick, self).preprocess_data(x)[0]
        noise = self.noise_dist.sample([x.size(0)])

        return x + 1 + noise,
        # return x + noise,

    def mean(self, etas):
        return torch.clamp(super().mean(etas) - 1 - self.noise_dist.mean, min=0., max=1.)
        # return torch.clamp(super().mean(etas) - self.noise_dist.mean, min=0., max=1.)

    def to_real_params(self, etas):
        return self.mean(etas),

    def impute(self, etas):
        probs = self.to_real_params(etas)[0]
        return (probs >= 0.5).float()

    def __str__(self):
        return f'{self.real_dist}*'


class PoissonGammaTrick(Gamma):
    def __init__(self):
        super().__init__()
        self.noise_dist = dist.Beta(1.1, 30)

    @property
    def real_dist(self) -> Base:
        return Poisson()

    def preprocess_data(self, x, mask=None):
        x = super().preprocess_data(x)[0]
        noise = self.noise_dist.sample([x.size(0)])

        return x + 1 + noise,

    def mean(self, etas):
        return super().mean(etas) - 1 - self.noise_dist.mean

    def to_real_params(self, etas):
        return torch.clamp(self.mean(etas), min=1e-10),  # rate > 0

    def impute(self, etas):
        rate = self.to_real_params(etas)[0]
        return rate.floor()

    def __str__(self):
        return f'{self.real_dist}*'


class Categorical(Base):
    def __init__(self, size):
        super().__init__()
        self.arg_constraints = [constraints.real_vector]
        self.size = size

    @property
    def dist(self):
        return dist.Categorical

    @property
    def parameters(self):
        return 'logits',

    @property
    def is_discrete(self):
        return True

    @property
    def real_parameters(self):
        return 'probs',

    @property
    def size_params(self):
        return [self.size]

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def impute(self, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).probs.max(dim=-1)[1]

    def params_from_data(self, x):
        new_x = to_one_hot(x, self.size)
        return probs_to_logits(new_x.sum(dim=0) / x.size(0)),

    def mean(self, etas):
        raise NotImplementedError()

    def to_params(self, etas):
        return etas[0],

    def to_real_params(self, etas):
        return logits_to_probs(self.to_params(etas)[0]),

    def __str__(self):
        return f'categorical({self.size})'


class CategoricalBernoulliTrick(Base):
    def __init__(self, size):
        super().__init__()
        del self._weight

        self.dists = [Bernoulli() for _ in range(size)]
        self.arg_constraints = reduce(list.__add__, [d.arg_constraints for d in self.dists])
        self.size = size

    @property
    def dist(self):
        return dist.Categorical

    @property
    def is_discrete(self):
        return True

    @property
    def real_parameters(self):
        return 'probs',

    @property
    def weight(self):
        return torch.tensor([d.weight for d in self.dists])

    @weight.setter
    def weight(self, value):
        assert self.num_dists == len(value)

        for d, v in zip(self.dists, value):
            d.weight = v

    def params_from_data(self, x):
        raise NotImplementedError()

    def to_params(self, etas):
        raise NotImplementedError()

    def real_params_from_data(self, x):
        return Categorical(self.size).real_params_from_data(x)

    @property
    def num_dists(self):
        return len(self.dists)

    @property
    def num_params(self):
        return sum([d.num_params for d in self.dists])

    def impute(self, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).probs.max(dim=-1)[1]

    def real_log_prob(self, x, etas):
        params = self.to_real_params(etas)
        params = dict(zip(self.real_parameters, params))

        new_x = x if len(x.size()) > 2 else to_one_hot(x, self.size)
        return dist.OneHotCategorical(**params).log_prob(new_x)

    def __getitem__(self, item):
        return self.dists[item]

    def preprocess_data(self, x, mask=None):
        new_x = super().preprocess_data(x)[0]
        x_one_hot = to_one_hot(new_x, self.size)

        new_x = []
        for i in range(self.size):
            new_x += self.dists[i].preprocess_data(x_one_hot[..., i])
        return new_x

    def scale_data(self, x, weight=None):
        return x

    def mean(self, etas):
        raise NotImplementedError()

    @property
    def f(self):
        return [lambda w: torch.tensor([1.0])]

    def to_real_params(self, etas):
        pos, probs = 0, []

        for i, d in enumerate(self.dists):
            probs.append(d.to_real_params(etas[pos: pos + d.num_params])[0])  # .detach())
            pos += d.num_params

        probs = torch.stack(probs, dim=-1)
        probs = torch.clamp(probs, min=1e-20)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs,

    def __str__(self):
        return f'categorical({self.size})+'


class CategoricalGammaTrick(CategoricalBernoulliTrick):
    def __init__(self, size):
        super(CategoricalGammaTrick, self).__init__(size)
        self.dists = [BernoulliGammaTrick() for _ in range(size)]

    @property
    def real_dist(self) -> Base:
        return CategoricalBernoulliTrick(self.size)

    @property
    def f(self):
        return reduce(list.__add__, [d.f for d in self.dists])

    @property
    def is_discrete(self):
        return False

    def __str__(self):
        return f'categorical({self.size})*'

