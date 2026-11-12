"""PyTorch port of joint FISTA video reconstruction, with spatial-only FFTs."""

import math
import warnings

import numpy as np
import torch
from scipy.fft import next_fast_len

try:
    from .backend import as_video, resolve_device, resolve_dtype
    from .priors import joint_prox
except ImportError:
    from backend import as_video, resolve_device, resolve_dtype
    from priors import joint_prox


class InverseSolver:
    """Solve .5||Ax-y||² + lam_spatial*R_s(x) + lam_temporal*R_t(x).

    Input layout remains (H,W,C,T). NumPy arrays or tensors are accepted;
    output is a tensor on the chosen device. The entire video is optimized
    jointly. intensity_constraint=True restricts reconstructed intensities to
    [0,1] inside optimization; the default preserves the original baseline.
    """

    def __init__(self, psf, algorithm="fista", num_iters=200, step_size=None,
                 lam_spatial=1e-3, lam_temporal=1e-3,
                 spatial_prior="wavelet", temporal_prior="tv",
                 wavelet="db4", wavelet_level=4, prox_iters=200, prox_tol=1e-5,
                 *, device="auto", dtype="auto", intensity_constraint=False):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        psf = torch.as_tensor(psf, device=self.device, dtype=self.dtype).detach()
        if psf.ndim != 2 or not all(psf.shape) or not torch.isfinite(psf).all().item():
            raise ValueError("psf must be a finite, nonempty 2D array")
        if (psf < 0).any().item() or psf.sum().item() <= 0:
            raise ValueError("psf must be nonnegative with positive sum")
        if algorithm not in ("fista", "ista", "gradient_descent"):
            raise ValueError("Unknown algorithm")
        if spatial_prior not in ("tv", "wavelet") or temporal_prior not in ("tv", "wavelet"):
            raise ValueError("Each prior must be 'tv' or 'wavelet'")
        if num_iters < 1 or not all(math.isfinite(v) and v >= 0 for v in (lam_spatial, lam_temporal)):
            raise ValueError("num_iters must be positive and lambdas finite/nonnegative")
        if step_size is not None and (not math.isfinite(step_size) or not 0 < step_size <= 1):
            raise ValueError("step_size must be in (0, 1] for the normalized PSF")
        if prox_iters < 1 or not math.isfinite(prox_tol) or prox_tol <= 0 or wavelet_level < 1:
            raise ValueError("Proximal iteration controls must be positive")
        self.h = psf / psf.sum()
        self.algorithm, self.num_iters = algorithm, num_iters
        self.step_size = 1.0 if step_size is None else step_size
        self.lam_spatial, self.lam_temporal = lam_spatial, lam_temporal
        self.spatial_prior, self.temporal_prior = spatial_prior, temporal_prior
        self.intensity_constraint = intensity_constraint
        self.prox_options = dict(wavelet=wavelet, level=wavelet_level,
                                 max_iters=prox_iters, tol=prox_tol,
                                 intensity_constraint=intensity_constraint)
        self._shape = None
        self.prox_diagnostics = []

    def _prepare(self, x):
        x = as_video(torch.as_tensor(x, dtype=self.dtype, device=self.device))
        if x.shape[:2] != self._shape:
            self._shape = x.shape[:2]
            self._fft_shape = tuple(next_fast_len(n + k - 1)
                                    for n, k in zip(self._shape, self.h.shape))
            self.H = torch.fft.fft2(self.h, s=self._fft_shape)[:, :, None, None]
            self._window = tuple(slice(k // 2, k // 2 + n)
                                 for n, k in zip(self._shape, self.h.shape))
        return x

    def _forward(self, x):
        spectrum = torch.fft.fft2(x, s=self._fft_shape, dim=(0, 1))
        return torch.fft.ifft2(spectrum * self.H, dim=(0, 1)).real[self._window]

    def _adjoint(self, y):
        embedded = y.new_zeros(self._fft_shape + tuple(y.shape[2:]))
        embedded[self._window] = y
        out = torch.fft.ifft2(torch.fft.fft2(embedded, dim=(0, 1)) * self.H.conj(),
                             dim=(0, 1)).real
        return out[:y.shape[0], :y.shape[1]]

    @torch.no_grad()
    def A(self, x):
        """Same zero-padded spatial convolution and same-size crop as CPU."""
        return self._forward(self._prepare(x))

    @torch.no_grad()
    def A_adjoint(self, y):
        """Exact transpose, including embedding the crop before correlation."""
        return self._adjoint(self._prepare(y))

    def prox(self, x):
        x = self._prepare(x)
        result, info = joint_prox(x, self.lam_spatial, self.lam_temporal,
                                  self.step_size, self.spatial_prior,
                                  self.temporal_prior, return_info=True,
                                  **self.prox_options)
        self.prox_diagnostics.append(info)
        return result

    @torch.no_grad()
    def __call__(self, y, callback=None, *, iterate_callback=None):
        """Reconstruct jointly; optionally observe each completed iterate.

        callback(iteration, data_loss) retains its existing interface.
        iterate_callback(iteration, data_loss, x) may read x for evaluation;
        it must not mutate the device tensor. Evaluation never feeds updates.
        """
        y = self._prepare(y)
        x = torch.zeros_like(y)
        v = x.clone()
        momentum = 1.0
        self.losses, self.prox_diagnostics = [], []
        for iteration in range(self.num_iters):
            base = v if self.algorithm == "fista" else x
            candidate = base - self.step_size * self._adjoint(self._forward(base) - y)
            new = candidate if self.algorithm == "gradient_descent" else self.prox(candidate)
            if self.algorithm == "gradient_descent" and self.intensity_constraint:
                new = new.clamp(0, 1)
            if self.algorithm == "fista":
                next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
                v = new + ((momentum - 1) / next_momentum) * (new - x)
                momentum = next_momentum
            x = new
            self.losses.append((0.5 * (self._forward(x) - y).square().sum()).item())
            if callback is not None:
                callback(iteration + 1, self.losses[-1])
            if iterate_callback is not None:
                iterate_callback(iteration + 1, self.losses[-1], x)
        missed = sum(not info["converged"] for info in self.prox_diagnostics)
        if missed:
            warnings.warn(f"{missed} proximal solves reached prox_iters before the gap tolerance; "
                          "increase prox_iters or relax prox_tol. See prox_diagnostics.",
                          RuntimeWarning, stacklevel=2)
        return x


def adjoint_test(solver, y_shape):
    # Match the original probe values; only these small test inputs originate
    # on CPU. All forward/adjoint work is on solver.device.
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=y_shape), device=solver.device, dtype=solver.dtype)
    y = torch.as_tensor(rng.normal(size=y_shape), device=solver.device, dtype=solver.dtype)
    return (torch.sum(y * solver.A(x)).item(),
            torch.sum(solver.A_adjoint(y) * x).item())
