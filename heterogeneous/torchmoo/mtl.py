import pdb
import math
from abc import ABCMeta, abstractmethod
from typing import Optional


import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from .utils import batch_product


def norm(tensor):
    return tensor.norm(p=2, dim=-1, keepdim=True)


def divide(numer, denom):
    """Numerically stable division"""
    epsilon = 1e-15
    return torch.sign(numer) * torch.sign(denom) * torch.exp(torch.log(numer.abs() + epsilon) - torch.log(denom.abs() + epsilon))


def unitary(tensor):
    return divide(tensor, norm(tensor) + 1e-15)


def projection(u, v):
    numer = torch.dot(u, v)
    denom = torch.dot(v, v)

    return numer / denom.clamp_min(1e-15) * v


class MOOMethod(nn.Module, metaclass=ABCMeta):

    requires_input: bool = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, grads: torch.Tensor, inputs: Optional[torch.Tensor],
                outputs: Optional[torch.Tensor]) -> torch.Tensor:
        """Computes the new task gradients based on the original ones.

        Given K gradients of size D, returns a new set of K gradients of size D based on some criterion.

        Args:
            grads: Tensor of size K x D with the different gradients.
            inputs: Tensor with the input of the forward pass (if ``requires_input`` is set to ``True``).
            outputs: Tensor with the K outputs of the module (_NOT_ used at the moment).

        Returns:
            A tensor of the same size as `grads` with the new gradients to use during backpropagation.
        """
        raise NotImplementedError('You need to implement the forward pass.')


class Compose(MOOMethod):
    def __init__(self, *modules):
        super().__init__()
        self.methods = nn.ModuleList(modules)
        self.requires_input = any([m.requires_input for m in modules])

    def forward(self, grads, inputs, outputs):
        for module in self.methods:
            grads = module(grads, inputs, outputs)

        return grads


class Identity(MOOMethod):
    def forward(self, grads: torch.Tensor, inputs: Optional[torch.Tensor],
                outputs: Optional[torch.Tensor]) -> torch.Tensor:
        return grads


class IMTLG(MOOMethod):
    requires_input = False

    def forward(self, grads, inputs, outputs):
        flatten_grads = grads.flatten(start_dim=1)
        num_tasks = len(grads)
        if num_tasks == 1:
            return grads

        grad_diffs, unit_diffs = [], []
        for i in range(1, num_tasks):
            grad_diffs.append(flatten_grads[0] - flatten_grads[i])
            unit_diffs.append(
                unitary(flatten_grads[0]) - unitary(flatten_grads[i]))
        grad_diffs = torch.stack(grad_diffs, dim=0)
        unit_diffs = torch.stack(unit_diffs, dim=0)

        DU_T = torch.einsum('ik,jk->ij', grad_diffs, unit_diffs)
        DU_T_inv = torch.pinverse(DU_T)  # DU_T could be non-invertible

        alphas = torch.einsum(
            'i,ki,kj->j', grads[0].flatten(), unit_diffs, DU_T_inv)
        alphas = torch.cat(
            (1-alphas.sum(dim=0).unsqueeze(dim=0), alphas), dim=0)

        return batch_product(grads, alphas)


class NSGD(MOOMethod):

    initial_grads: torch.Tensor
    requires_input = False

    def __init__(self, num_tasks, update_at=20):
        super(NSGD, self).__init__()

        self.num_tasks = num_tasks
        self.update_at = update_at
        self.register_buffer('initial_grads', torch.ones(num_tasks))
        self.counter = 0

    def forward(self, grads, inputs, outputs):
        grad_norms = grads.flatten(start_dim=1).norm(dim=1)

        if self.initial_grads is None or self.counter == self.update_at:
            self.initial_grads = grad_norms

        self.counter += 1

        conv_ratios = grad_norms / self.initial_grads.clamp_min(1e-15)
        alphas = conv_ratios / conv_ratios.sum().clamp_min(1e-15)
        alphas = alphas / alphas.sum()  # TODO does this help?

        weighted_sum_norms = (alphas * grad_norms).sum()
        # weighted_sum_norms = grad_norms.sum() / len(grad_norms)
        grads = batch_product(grads, weighted_sum_norms /
                              grad_norms.clamp_min(1e-15))
        return grads

        # grad_norms = [torch.norm(g, keepdim=True) for g in grads]
        # mean_norm = sum(grad_norms) / len(grad_norms)
        # if self.initial_grads is None or self.counter == self.update_at:
        #     self.initial_grads = grad_norms
        #
        # self.counter += 1
        #
        # # norm_grad_norms = [x / (y + 1e-5) for x, y in zip(grad_norms, self.initial_grads)]
        # # norm_mean_norm = sum(norm_grad_norms) / len(norm_grad_norms)
        # grad_norms = [x / (y + 1e-5) for x, y in zip(grad_norms, self.initial_grads)]
        # mean_norm = sum(grad_norms) / len(grad_norms)
        #
        # # grads = [g * (mean_norm / (n + 1e-5)) for g, n in zip(grads, grad_norms)]
        # grads = [g * (self.num_tasks * n / sum(grad_norms) ) for g, n in zip(grads, grad_norms)]
        # # alphas = torch.softmax(torch.stack(norm_grad_norms, dim=0), dim=0).unbind(dim=0)
        # # alphas = [(nn / (norm_mean_norm + 1e-5)) for nn in norm_grad_norms]
        # # grads = [g * (mean_norm / (n + 1e-5)) * nn for g, [n, nn] in zip(grads, zip(grad_norms, alphas))]
        # return grads


# Mostly based on the original paper description and this implementation
# https://github.com/tensorflow/lingvo/blob/master/lingvo/core/graddrop.py ( Latest commit ae4d22f on Feb 25 )

class GradDrop(MOOMethod):
    requires_input = True

    def __init__(self, leakage):
        super(GradDrop, self).__init__()
        assert all([x >= 0 for x in leakage]), "all leakages should be in the range [0, 1]"
        assert all([x <= 1 for x in leakage]), "all leakages should be in the range [0, 1]"
        self.leakage = leakage

    def forward(self, grads, inputs, outputs):
        assert len(self.leakage) == len(grads), "leakage parameters should equate incoming task gradients"
        sign_grads = [None for _ in range(len(grads))]
        for i in range(len(grads)):
            sign_grads[i] = inputs.sign() * grads[i]
            if len(grads[0].size()) > 1:  # It is batch-separated
                sign_grads[i] = grads[i].sum(dim=0, keepdim=True)

        odds = 0.5 * (1 + sum(sign_grads) / (sum(map(torch.abs, sign_grads)) + 1e-15))
        assert odds.size() == sign_grads[0].size()

        new_grads = []
        samples = torch.rand(odds.size(), device=grads[0].device)
        for i in range(len(grads)):
            # The paper describes an monotonically increasing function, odd around (0.5, 0.5) and that maps
            # [0, 1] -> [0, 1]. The use as an example the identity function, which does not hold the odd assumption
            mask_i = (odds > samples) * \
                (sign_grads[i] > 0) + (odds < samples) * (sign_grads[i] < 0)

            mask_i = self.leakage[i] + ((1-self.leakage[i]) * mask_i)
            assert mask_i.size() == odds.size()
            new_grads.append(mask_i * grads[i])

        return torch.stack(new_grads, dim=0)


class GradNormBase(MOOMethod):

    initial_values: torch.Tensor
    counter: torch.Tensor

    def __init__(self, num_tasks, alpha, update_at=20):
        super(GradNormBase, self).__init__()

        self.epsilon = 1e-5
        self.num_tasks = num_tasks
        self.weight_ = nn.Parameter(
            torch.ones([num_tasks]), requires_grad=True)
        self.alpha = alpha

        self.update_at = update_at
        self.register_buffer('initial_values', torch.ones(self.num_tasks))
        self.register_buffer('counter', torch.zeros([]))

    @property
    def weight(self):
        ws = self.weight_.exp().clamp_min(self.epsilon)
        norm_coef = self.num_tasks / ws.sum()
        # norm_coef = 1. / ws.sum()
        return ws * norm_coef

    def _forward(self, grads, values):
        # Update the initial values if necessary
        if self.initial_values is None or self.counter == self.update_at:
            self.initial_values = torch.tensor(values)
        self.counter += 1

        with torch.enable_grad():
            # Compute the norm of the individual and average gradients
            grads_norm = grads.flatten(start_dim=1).norm(p=2, dim=1)
            mean_grad_norm = torch.mean(batch_product(
                grads_norm, self.weight), dim=0).detach().clone()

            # Normalized values by dividing for the initial ones (at step 0 of training)
            values = [x / y.clamp_min(self.epsilon)
                      for x, y in zip(values, self.initial_values)]
            average_value = sum(values) / len(values)

            loss = grads.new_zeros([])
            for i, [grad, value] in enumerate(zip(grads_norm, values)):
                r_i = value / average_value.clamp_min(self.epsilon)
                loss += torch.abs(grad * self.weight[i] -
                                  mean_grad_norm * r_i.pow(self.alpha))
            loss.backward()

        new_grads = batch_product(grads, self.weight.detach())
        return new_grads


class GradNorm(GradNormBase):
    requires_input = False

    # @typechecked
    def forward(self, grads, inputs, outputs: torch.Tensor):
        return self._forward(grads, outputs)


class GradNormModified(GradNormBase):
    r"""
    Proposed by us, instead of using the task loss convergence, we use the task-gradient convergence
    """

    requires_input = False

    def forward(self, grads, inputs, outputs):
        return self._forward(grads, grads.flatten(start_dim=1).norm(p=2, dim=1))


class PCGrad(MOOMethod):
    requires_input = False

    def forward(self, grads, inputs, outputs):
        size = grads.size()[1:]
        num_tasks = grads.size(0)

        # cos_sim_ij = grads[0].new_ones((num_tasks, num_tasks))
        grads_list = [g.flatten() for g in grads]  # Eq. 1 Original paper

        # Precompute cosine similarity  (as defined in the original paper, Section 3)
        # for i in range(num_tasks):
        #     cos_sim_ij[i, i] = 1.
        #     for j in range(i + 1, num_tasks):
        #         cos_sim_ij[i, j] = torch.cosine_similarity(grads_list[i], grads_list[j], dim=0)
        #         cos_sim_ij[j, i] = cos_sim_ij[i, j]

        # Randomly project gradients
        new_grads = [None for _ in range(num_tasks)]
        for i in np.random.permutation(num_tasks):
            grad_i = grads_list[i]
            for j in np.random.permutation(num_tasks):
                if i == j:
                    continue

                grad_j = grads_list[j]
                # if cos_sim_ij[i, j] < 0:
                if torch.cosine_similarity(grad_i, grad_j, dim=0) < 0:
                    grad_i = grad_i - projection(grad_i, grad_j)
                    assert id(grads_list[i]) != id(grad_i), 'Aliasing!'

            new_grads[i] = grad_i.reshape(size)

        return torch.stack(new_grads, dim=0)


class GradVac(MOOMethod):
    requires_input = False

    def __init__(self, decay):
        super(GradVac, self).__init__()
        self.decay = decay

    def forward(self, grads, inputs, outputs):
        def vac_projection(u, v, pre_ema, post_ema):
            norm_u = torch.dot(u, u).sqrt()
            norm_v = torch.dot(v, v).sqrt()

            numer = norm_u * (pre_ema * math.sqrt(1 - post_ema **
                              2) - post_ema * math.sqrt(1 - pre_ema**2))
            denom = norm_v * math.sqrt(1 - pre_ema**2)

            return numer / denom.clamp_min(1e-15) * v

        size = grads.size()[1:]
        num_tasks = grads.size(0)

        grads_list = [g.flatten() for g in grads]  # Eq. 1 Original paper
        ema = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]

        # Randomly project gradients
        new_grads = []
        for i in range(num_tasks):
            grad_i = grads_list[i]
            for j in np.random.permutation(num_tasks):
                if i == j:
                    continue

                grad_j = grads_list[j]
                cos_sim = torch.cosine_similarity(grad_i, grad_j, dim=0)
                if cos_sim < ema[i][j]:
                    grad_i = grad_i + \
                        vac_projection(grad_i, grad_j, ema[i][j], cos_sim)
                    assert id(grads_list[i]) != id(grad_i), 'Aliasing!'
                ema[i][j] = (1 - self.decay) * ema[i][j] + self.decay * cos_sim

            new_grads.append(grad_i.reshape(size))

        return torch.stack(new_grads, dim=0)


# Code taken from the official repository of MGDA-UB:
# https://github.com/isl-org/MultiObjectiveOptimization

class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = float('inf')
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k], vecs[j][k]).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k], vecs[i][k]).item()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).item()
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().item()
                            for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * \
                np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn


class MGDAUB(MOOMethod):
    requires_input = False

    def forward(self, grads, inputs, outputs):
        epsilon = 1e-3
        shape = grads.size()[1:]
        grads = grads.flatten(start_dim=1).unsqueeze(
            dim=1)  # TODO temporal patch

        weights, min_norm = MinNormSolver.find_min_norm_element(
            grads.unbind(dim=0))
        # weights, min_norm = MinNormSolver.find_min_norm_element_FW(grads)
        assert len(weights) == len(grads)

        weights = [min(w, epsilon) for w in weights]

        grads = torch.stack([g.reshape(shape) * w for g,
                            w in zip(grads, weights)], dim=0)
        return grads


class CAGrad(MOOMethod):
    requires_input = False

    def __init__(self, alpha):
        super(CAGrad, self).__init__()
        self.alpha = alpha

    def forward(self, grads, inputs, outputs):  # grads = (num_tasks x something...)
        shape = grads.size()  # [1:]
        num_tasks = len(grads)
        grads = grads.flatten(start_dim=1).t()

        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(num_tasks) / num_tasks
        bnds = tuple((0, 1) for _ in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

        A = GG.numpy()
        b = x_start.copy()
        c = (self.alpha*g0_norm+1e-8).item()

        def objfn(x):
            return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
                x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x

        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        # g = grads.mean(1) + lmbda * gw
        g = (grads + lmbda * gw.unsqueeze(1)) / num_tasks

        g = g.t().reshape(shape)
        # grads = torch.stack([g / num_tasks for _ in range(num_tasks)], dim=0)
        grads = g

        return grads
