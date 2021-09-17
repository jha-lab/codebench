import numpy as np
from scipy.stats import norm
import torch
import sys

# Different acquisition functions
def gosh_acq(prediction, std, explore_type='ucb'):

    prediction = torch.Tensor(prediction)
    std = torch.Tensor(std)

    # Upper confidence bound (UCB) acquisition function
    if explore_type == 'ucb':
        explore_factor = 0.5
        obj = prediction - explore_factor * std

    # Purely uncertainty based sampling
    elif explore_type == 'unc':
        obj = std

    else:
        raise NotImplementedError(f'{explore_type} is not supported')

    return obj