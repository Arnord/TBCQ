#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import torch.nn as nn

from typing import Optional, Union
from .func import weighted_linear, normal_differential_sample

from torch.distributions.multivariate_normal import MultivariateNormal, _precision_to_scale_tril, _batch_mahalanobis

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
from torch.distributions.constraints import Constraint


class ValStepSchedule:
    def __init__(self, optimizer, lr_scheduler_nstart, lr_scheduler_nepochs, lr_scheduler_factor, logger):
        self.optimizer = optimizer
        self.lr_scheduler_nstart = lr_scheduler_nstart
        self.lr_scheduler_nepochs = lr_scheduler_nepochs
        self.lr_scheduler_factor = lr_scheduler_factor
        self.logger = logger

    def step(self, all_val_metrices, val_metric):
        if len(all_val_metrices) > self.lr_scheduler_nepochs and \
                val_metric >= max(all_val_metrices[int(-self.lr_scheduler_nepochs - 1):-1]):

            # adapt new learning rate in the optimizer
            for param in self.optimizer.param_groups:
                param['lr'] = param['lr'] / self.lr_scheduler_factor


class PeriodSchedule:
    def __init__(self, scheduler, periods, logger):
        self.scheduler = scheduler
        self.round = 0
        self.periods = periods
        self.logger = logger

    def step(self, *args, **kwargs):
        self.round += 1
        if self.round == self.periods:
            self.round = 0
            self.logger('Updating learning rate')
            self.scheduler.step(*args, **kwargs)
        else:
            pass


class DBlock(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b

class DBlock_Relu(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock_Relu, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        mu = self.fc_mu(t)
        logsigma = torch.relu(self.fc_logsigma(t))
        return mu, logsigma


class DBlock_Relu_Mini(nn.Module):
    """ A basie building block for parametralize a normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock_Relu_Mini, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = self.fc2(t)
        mu = self.fc_mu(t)
        logsigma = torch.relu(self.fc_logsigma(t))
        return mu, logsigma


class PreProcess(nn.Module):

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = self.fc2(t)
        return t


class MLP(nn.Module):
    def __init__(self, input_size, hidden_num, out_size, num_mlp_layers, nonlinearity=torch.nn.Tanh):
        super(MLP, self).__init__()
        L = nn.ModuleList()
        L.append(nn.Linear(input_size, hidden_num))
        L.append(nonlinearity())
        for _ in range(num_mlp_layers-1):
            L.append(nn.Linear(hidden_num, hidden_num))
            L.append(nonlinearity())
        L.append(nn.Linear(hidden_num, out_size))
        self.mlp = nn.Sequential(
            *L
        )

    def forward(self, x):
        return self.mlp(x)



class _DiagPositiveDefinite(Constraint):
    """
    Constrain to positive-definite diagonal matrices.
    """

    def check(self, value):
        return value.diagonal(dim1=-2, dim2=-1).min(dim=-1)[0] > 0.0


diag_positive_definite = _DiagPositiveDefinite()


class DiagMultivariateNormal(torch.distributions.multivariate_normal.MultivariateNormal):

    arg_constraints = {'loc': constraints.real_vector,
                       # positive_definite.positive_definite is replaced by diag_positive_definite
                       'covariance_matrix': diag_positive_definite,
                       'precision_matrix': diag_positive_definite,
                       'scale_tril': constraints.lower_cholesky}

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):

        if torch.all(covariance_matrix==0):
            covariance_matrix = torch.diag_embed(covariance_matrix.diagonal(dim1=-2, dim2=-1) + 1e-9)

        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")

        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix, loc_ = torch.broadcast_tensors(precision_matrix, loc_)
        self.loc = loc_[..., 0]  # drop rightmost dim

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            #self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
            self._unbroadcasted_scale_tril = torch.sqrt(covariance_matrix)
        else:  # precision_matrix is not None
            raise NotImplementedError('Only covariance_matrix or scale_tril may be specified')


def softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(1 + torch.exp(x)), x
    )


def sqrt_softplus(x, threshold=20):
    return x.exp().sqrt()


def inverse_softplus(x, threshold=20):
    return torch.where(
        x < threshold, torch.log(torch.exp(x) - torch.ones_like(x)), x
    )


def logsigma2cov(logsigma):
    return torch.diag_embed(softplus(logsigma) ** 2)


def sqrt_logsigma2cov(logsigma):
    return torch.diag_embed(sqrt_softplus(logsigma))

def cov2logsigma(cov):
    return inverse_softplus(torch.sqrt(cov.diagonal(dim1=-2, dim2=-1)))


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val

def merge_first_two_dims(tensor):
    size = tensor.size()
    return tensor.contiguous().reshape(-1, *size[2:])


def split_first_dim(tensor, sizes=None):
    if sizes is None:
        sizes = tensor.size()[0]
    import numpy
    assert type(sizes) is tuple and numpy.prod(sizes) == tensor.size()[0]
    return tensor.contiguous().reshape(*sizes, *tensor.size()[1:])


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


# _kid_gauss_sample
def _kid_gauss(self, mean_1, std_1, mean_2, std_2):
    EPS = torch.finfo(torch.float64).eps
    kid_element = (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)

    return 0.5 * torch.sum(kid_element)


CONST_EPS = 1e-10


def orthogonal_initWeights(
    net: nn.Module,
    ) -> None:
    for e in net.parameters():
        if len(e.size()) >= 2:
            nn.init.orthogonal_(e)


def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)
