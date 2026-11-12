"""GPU joint spatial/temporal prox, matching the original NumPy equations."""

from functools import lru_cache
import math

import pywt
import torch
import torch.nn.functional as F

try:
    from .backend import as_video
except ImportError:
    from backend import as_video


def soft_thresh(x, threshold):
    return x.sign() * (x.abs() - threshold).clamp_min(0)


@lru_cache(maxsize=128)
def _filter_bank(wavelet, even, device, dtype):
    # PyWavelets supplies only constant filter coefficients. No video data is
    # transformed on the CPU. Reversal converts convolution to correlation.
    basis = pywt.Wavelet(wavelet)
    filters = torch.tensor([basis.dec_lo[::-1], basis.dec_hi[::-1]],
                           device=device, dtype=dtype).unsqueeze(1)
    length = filters.shape[-1]
    indices = (torch.arange(even + length - 2, device=device)
               - (length // 2 - 1)) % even
    return filters, indices


@torch.no_grad()
def _wavelet(x, axes, wavelet, level, inverse=False):
    """Same square, separable periodized transform and odd-tail convention.

    Each axis is a batch of independent 1D signals. The transpose of periodic
    extension folds the synthesized samples back into the even prefix.
    """
    out = x.clone()
    stages = []
    for axis in axes:
        size = x.shape[axis]
        for _ in range(level):
            if size - size % 2 < 2:
                break
            stages.append((axis, size))
            size = (size + 1) // 2
    for axis, size in reversed(stages) if inverse else stages:
        even = size - size % 2
        coarse = (size + 1) // 2
        view = out.movedim(axis, -1)
        filters, indices = _filter_bank(wavelet, even, x.device, x.dtype)
        batch_shape = view.shape[:-1]
        if inverse:
            coefficients = torch.stack((view[..., :even // 2],
                                        view[..., coarse:size]), dim=-2)
            extended = F.conv_transpose1d(coefficients.reshape(-1, 2, even // 2),
                                         filters, stride=2).squeeze(1)
            restored = x.new_zeros((extended.shape[0], even))
            restored.index_add_(1, indices, extended)
            values = torch.cat((restored.reshape(*batch_shape, even),
                                view[..., even // 2:coarse]), dim=-1)
        else:
            extended = view[..., :even].index_select(-1, indices)
            coefficients = F.conv1d(extended.reshape(-1, 1, indices.numel()),
                                    filters, stride=2).reshape(*batch_shape, 2, even // 2)
            values = torch.cat((coefficients[..., 0, :], view[..., even:size],
                                coefficients[..., 1, :]), dim=-1)
        view[..., :size] = values
    return out


def _difference_adjoint(p, axis, shape):
    out = p.new_zeros(shape)
    left = [slice(None)] * len(shape)
    right = left.copy()
    left[axis], right[axis] = slice(None, -1), slice(1, None)
    out[tuple(left)] -= p
    out[tuple(right)] += p
    return out


def _operators(shape, prior, axes, weight, wavelet, level):
    axes = tuple(axis for axis in axes if shape[axis] > 1)
    if weight == 0 or not axes:
        return []
    if prior == "wavelet":
        return [(lambda x: _wavelet(x, axes, wavelet, level),
                 lambda p: _wavelet(p, axes, wavelet, level, inverse=True),
                 weight, 1.0)]
    return [(lambda x, axis=axis: torch.diff(x, dim=axis),
             lambda p, axis=axis: _difference_adjoint(p, axis, shape),
             weight, 4.0) for axis in axes]


@torch.no_grad()
def joint_prox(x, lam_spatial, lam_temporal, step_size,
               spatial_prior="wavelet", temporal_prior="tv", *,
               wavelet="db4", level=4, max_iters=200, tol=1e-5,
               return_info=False, intensity_constraint=False):
    """Accelerated projected dual solve of the SUM of the two penalties.

    With intensity_constraint, solve over 0 <= result <= 1. Projection is
    part of the dual gradient and primal recovery, so the box and both priors
    are optimized jointly. Measurements are never clipped. Otherwise matches
    the original unconstrained CPU method.
    Returns a tensor on the input device, without an autograd graph.
    """
    x = as_video(x)
    if spatial_prior not in ("tv", "wavelet") or temporal_prior not in ("tv", "wavelet"):
        raise ValueError("Each prior must be 'tv' or 'wavelet'")
    if not all(math.isfinite(v) and v >= 0 for v in (lam_spatial, lam_temporal, step_size)):
        raise ValueError("Lambdas and step_size must be finite and nonnegative")
    if max_iters < 1 or level < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("max_iters, level, and tol must be positive")
    if not pywt.Wavelet(wavelet).orthogonal:
        raise ValueError("The wavelet must be orthogonal")
    ops = (_operators(x.shape, spatial_prior, (0, 1), step_size * lam_spatial, wavelet, level)
           + _operators(x.shape, temporal_prior, (3,), step_size * lam_temporal, wavelet, level))
    info = {"iterations": 0, "gap": 0.0, "converged": True}
    if not ops:
        result = x.clamp(0, 1) if intensity_constraint else x.clone()
        return (result, info) if return_info else result
    dual = [torch.zeros_like(forward(x)) for forward, _, _, _ in ops]
    extrapolated = [p.clone() for p in dual]
    sigma = 1.0 / sum(bound for _, _, _, bound in ops)
    momentum = 1.0
    for iteration in range(max_iters):
        estimate = x.clone()
        for p, (_, adjoint, _, _) in zip(extrapolated, ops):
            estimate -= adjoint(p)
        if intensity_constraint:
            estimate.clamp_(0, 1)
        updated = [(q + sigma * forward(estimate)).clamp(-weight, weight)
                   for q, (forward, _, weight, _) in zip(extrapolated, ops)]
        result = x.clone()
        for p, (_, adjoint, _, _) in zip(updated, ops):
            result -= adjoint(p)
        if intensity_constraint:
            result.clamp_(0, 1)
        # result minimizes the quadratic plus box indicator at this dual
        # point. Their Fenchel gap is zero; only the prior gaps remain.
        gap, penalty = x.new_zeros(()), x.new_zeros(())
        for p, (forward, _, weight, _) in zip(updated, ops):
            coefficients = forward(result)
            weighted = weight * coefficients.abs()
            penalty += weighted.sum()
            gap += (weighted - p * coefficients).sum()
        primal = 0.5 * (result - x).square().sum() + penalty
        # Only scalar diagnostics cross to the host, once per inner iteration.
        gap_value, primal_value = torch.stack((gap, primal)).tolist()
        converged = gap_value <= tol * max(primal_value, 1e-12)
        info = {"iterations": iteration + 1, "gap": max(gap_value, 0.0),
                "converged": converged}
        if converged:
            break
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = [new + ((momentum - 1) / next_momentum) * (new - old)
                        for new, old in zip(updated, dual)]
        dual, momentum = updated, next_momentum
    return (result, info) if return_info else result


def prox_tv(x, lam, step_size, *, temporal=False, **kwargs):
    return joint_prox(x, 0 if temporal else lam, lam if temporal else 0,
                      step_size, "tv", "tv", **kwargs)


def prox_wavelet(x, lam, step_size, *, temporal=False, **kwargs):
    return joint_prox(x, 0 if temporal else lam, lam if temporal else 0,
                      step_size, "wavelet", "wavelet", **kwargs)
