"""Joint spatial/temporal proximal operators for (H, W, C, T) videos."""

import numpy as np
import pywt


def soft_thresh(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def _wavelet(x, axes, wavelet, level, inverse=False):
    """Square orthonormal separable transform, including odd-sized inputs.

    Each level transforms the even prefix and carries an unmatched last sample
    into the next coarse level. This avoids redundant padding, so the inverse
    is the adjoint. Periodization applies within each even prefix.
    """
    out = np.array(x, dtype=float, copy=True)
    stages = []
    for axis in axes:
        size = x.shape[axis]
        for _ in range(level):
            even = size - size % 2
            if even < 2:
                break
            stages.append((axis, size))
            size = (size + 1) // 2
    for axis, size in reversed(stages) if inverse else stages:
        even = size - size % 2
        coarse = (size + 1) // 2
        view = np.moveaxis(out, axis, 0)
        if inverse:
            restored = pywt.idwt(view[:even // 2], view[coarse:size],
                                wavelet, mode="periodization", axis=0)
            view[:size] = np.concatenate((restored, view[even // 2:coarse]), axis=0)
        else:
            low, high = pywt.dwt(view[:even], wavelet,
                                mode="periodization", axis=0)
            view[:size] = np.concatenate((low, view[even:size], high), axis=0)
    return out


def _difference_adjoint(p, axis, shape):
    out = np.zeros(shape, dtype=float)
    left = [slice(None)] * len(shape)
    right = left.copy()
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
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
    return [(lambda x, axis=axis: np.diff(x, axis=axis),
             lambda p, axis=axis: _difference_adjoint(p, axis, shape),
             weight, 4.0) for axis in axes]


def joint_prox(x, lam_spatial, lam_temporal, step_size,
               spatial_prior="wavelet", temporal_prior="tv", *,
               wavelet="db4", level=4, max_iters=200, tol=1e-5,
               return_info=False):
    """Prox of the SUM of the selected priors via an accelerated dual solve.

    Minimize .5||z-x||² + step*(lam_spatial*R_s(z)+lam_temporal*R_t(z)).
    Stop on a relative primal-dual gap. TV uses anisotropic L1 differences
    with open boundaries. Wavelets penalize all coefficients, as in Lab 8.
    Channels are independent. T=1 disables the temporal penalty. There is
    no implicit positivity constraint. max_iters bounds inner-solve accuracy.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 4 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("Expected a finite, nonempty (H, W, C, T) array")
    if spatial_prior not in ("tv", "wavelet") or temporal_prior not in ("tv", "wavelet"):
        raise ValueError("Each prior must be 'tv' or 'wavelet'")
    if not all(np.isfinite(v) and v >= 0 for v in (lam_spatial, lam_temporal, step_size)):
        raise ValueError("Lambdas and step_size must be finite and nonnegative")
    if max_iters < 1 or level < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iters, level, and tol must be positive")
    if not pywt.Wavelet(wavelet).orthogonal:
        raise ValueError("The wavelet must be orthogonal")
    ops = (_operators(x.shape, spatial_prior, (0, 1), step_size * lam_spatial, wavelet, level)
           + _operators(x.shape, temporal_prior, (3,), step_size * lam_temporal, wavelet, level))
    info = {"iterations": 0, "gap": 0.0, "converged": True}
    if not ops:
        return (x.copy(), info) if return_info else x.copy()
    dual = [np.zeros_like(forward(x)) for forward, _, _, _ in ops]
    extrapolated = [p.copy() for p in dual]
    sigma = 1.0 / sum(bound for _, _, _, bound in ops)
    momentum = 1.0
    for iteration in range(max_iters):
        estimate = x.copy()
        for p, (_, adjoint, _, _) in zip(extrapolated, ops):
            estimate -= adjoint(p)
        updated = [np.clip(q + sigma * forward(estimate), -weight, weight)
                   for q, (forward, _, weight, _) in zip(extrapolated, ops)]
        result = x.copy()
        for p, (_, adjoint, _, _) in zip(updated, ops):
            result -= adjoint(p)
        gap, penalty = 0.0, 0.0
        for p, (forward, _, weight, _) in zip(updated, ops):
            coefficients = forward(result)
            penalty += weight * np.abs(coefficients).sum()
            gap += (weight * np.abs(coefficients) - p * coefficients).sum()
        primal = 0.5 * np.square(result - x).sum() + penalty
        converged = gap <= tol * max(primal, 1e-12)
        info = {"iterations": iteration + 1, "gap": float(max(gap, 0)),
                "converged": bool(converged)}
        if converged:
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = [new + ((momentum - 1) / next_momentum) * (new - old)
                        for new, old in zip(updated, dual)]
        dual, momentum = updated, next_momentum
    return (result, info) if return_info else result


def prox_tv(x, lam, step_size, *, temporal=False, **kwargs):
    """TV prox over spatial axes, or only time when temporal=True."""
    return joint_prox(x, 0 if temporal else lam, lam if temporal else 0,
                      step_size, "tv", "tv", **kwargs)


def prox_wavelet(x, lam, step_size, *, temporal=False, **kwargs):
    """Wavelet prox over spatial axes, or only time when temporal=True."""
    return joint_prox(x, 0 if temporal else lam, lam if temporal else 0,
                      step_size, "wavelet", "wavelet", **kwargs)
