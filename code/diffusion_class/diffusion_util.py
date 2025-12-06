
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import enum
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


def mean_flat(x):
    """
    Computes the mean of the tensor `x` over all dimensions except the first (typically the batch dimension).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, ...)

    Returns:
        torch.Tensor: A 1D tensor containing the mean values for each sample in the batch.
    """
    return x.mean(dim=list(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
        Compute the KL divergence between two gaussians.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
        A fast approximation of the cumulative distribution function of the standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) *
                         (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
    :param x: the target images. It is assumed that this was uint8 values, rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(
                cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
        Get a pre-defined beta schedule for the given name.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Specifies the type of output the model predicts.
    """
    # enum.auto() is a helper function in Python's enum module that automatically assigns values 
    # to enum members. The values start at 1 and increment by 1 for each subsequent member.
    PREVIOUS_X = enum.auto()       # The model predicts x_{t-1}
    START_X = enum.auto()          # The model predicts x_0
    EPSILON = enum.auto()          # The model predicts epsilon
    MOLLIFIED_EPSILON = enum.auto()  # The model predicts mollified epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    """

    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss
    L1 = enum.auto()  # use raw L1 loss


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
