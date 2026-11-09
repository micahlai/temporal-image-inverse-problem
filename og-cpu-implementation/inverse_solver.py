"""Joint video reconstruction with a spatial-only, zero-boundary PSF model."""

import warnings

import numpy as np
from scipy.fft import fft2, ifft2, next_fast_len

try:
    from .priors import joint_prox
except ImportError:
    from priors import joint_prox


class InverseSolver:
    """Solve .5||Ax-y||² + lam_spatial*R_s(x) + lam_temporal*R_t(x).

    Input and output: (H, W, C, T), i.e. (Y, X, channel, time) in NumPy
    image conventions. A single nonnegative 2D PSF is normalized to unit
    sum and shared by all channels/frames. Origin: (kh//2, kw//2).
    Spatial convolution has zero boundaries. Padding is internal to A;
    unlike Lab 8, the optimization variable is the video itself.
    """

    def __init__(self, psf, algorithm="fista", num_iters=200, step_size=None,
                 lam_spatial=1e-3, lam_temporal=1e-3,
                 spatial_prior="wavelet", temporal_prior="tv",
                 wavelet="db4", wavelet_level=4, prox_iters=200, prox_tol=1e-5):
        psf = np.asarray(psf, dtype=float)
        if psf.ndim != 2 or not all(psf.shape) or not np.isfinite(psf).all():
            raise ValueError("psf must be a finite, nonempty 2D array")
        if np.any(psf < 0) or psf.sum() <= 0:
            raise ValueError("psf must be nonnegative with positive sum")
        if algorithm not in ("fista", "ista", "gradient_descent"):
            raise ValueError("Unknown algorithm")
        if spatial_prior not in ("tv", "wavelet") or temporal_prior not in ("tv", "wavelet"):
            raise ValueError("Each prior must be 'tv' or 'wavelet'")
        if num_iters < 1 or not all(np.isfinite(v) and v >= 0 for v in (lam_spatial, lam_temporal)):
            raise ValueError("num_iters must be positive and lambdas finite/nonnegative")
        # Unit-sum nonnegative convolution has ||A|| <= 1, including crops.
        # Thus L=1 safely bounds A* A, replacing Lab 8's estimate of A alone.
        if step_size is not None and (not np.isfinite(step_size) or not 0 < step_size <= 1):
            raise ValueError("step_size must be in (0, 1] for the normalized PSF")
        if prox_iters < 1 or not np.isfinite(prox_tol) or prox_tol <= 0 or wavelet_level < 1:
            raise ValueError("Proximal iteration controls must be positive")
        self.h = psf / psf.sum()
        self.algorithm, self.num_iters = algorithm, num_iters
        self.step_size = 1.0 if step_size is None else step_size
        self.lam_spatial, self.lam_temporal = lam_spatial, lam_temporal
        self.spatial_prior, self.temporal_prior = spatial_prior, temporal_prior
        self.prox_options = dict(wavelet=wavelet, level=wavelet_level,
                                 max_iters=prox_iters, tol=prox_tol)
        self._shape = None
        self.prox_diagnostics = []

    def _prepare(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 4 or not all(x.shape) or not np.isfinite(x).all():
            raise ValueError("Expected a finite, nonempty (H, W, C, T) video")
        if x.shape[:2] != self._shape:
            self._shape = x.shape[:2]
            self._fft_shape = tuple(next_fast_len(n + k - 1)
                                    for n, k in zip(self._shape, self.h.shape))
            self.H = fft2(self.h, s=self._fft_shape)[:, :, None, None]
            self._window = tuple(slice(k // 2, k // 2 + n)
                                 for n, k in zip(self._shape, self.h.shape))
        return x

    def A(self, x):
        """Spatial linear convolution, cropped to the input video dimensions."""
        x = self._prepare(x)
        spectrum = fft2(x, s=self._fft_shape, axes=(0, 1))
        return ifft2(spectrum * self.H, axes=(0, 1)).real[self._window]

    def A_adjoint(self, y):
        """Exact transpose of the zero-padded/cropped forward convolution."""
        y = self._prepare(y)
        embedded = np.zeros(self._fft_shape + y.shape[2:], dtype=float)
        embedded[self._window] = y
        out = ifft2(fft2(embedded, axes=(0, 1)) * self.H.conj(), axes=(0, 1)).real
        return out[:y.shape[0], :y.shape[1]]

    def prox(self, x):
        result, info = joint_prox(x, self.lam_spatial, self.lam_temporal,
                                  self.step_size, self.spatial_prior,
                                  self.temporal_prior, return_info=True,
                                  **self.prox_options)
        self.prox_diagnostics.append(info)
        return result

    def __call__(self, y, callback=None):
        """Reconstruct the whole video jointly; callback(iteration, data_loss)."""
        y = self._prepare(y)
        x = np.zeros_like(y)
        v = x.copy()
        momentum = 1.0
        self.losses, self.prox_diagnostics = [], []
        for iteration in range(self.num_iters):
            base = v if self.algorithm == "fista" else x
            candidate = base - self.step_size * self.A_adjoint(self.A(base) - y)
            new = candidate if self.algorithm == "gradient_descent" else self.prox(candidate)
            if self.algorithm == "fista":
                next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
                v = new + ((momentum - 1) / next_momentum) * (new - x)
                momentum = next_momentum
            x = new
            self.losses.append(float(0.5 * np.square(self.A(x) - y).sum()))
            if callback is not None:
                callback(iteration + 1, self.losses[-1])
        missed = sum(not info["converged"] for info in self.prox_diagnostics)
        if missed:
            warnings.warn(f"{missed} proximal solves reached prox_iters before the gap tolerance; "
                          "increase prox_iters or relax prox_tol. See prox_diagnostics.",
                          RuntimeWarning, stacklevel=2)
        return x


def adjoint_test(solver, y_shape):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=y_shape), rng.normal(size=y_shape)
    return np.vdot(y, solver.A(x)), np.vdot(solver.A_adjoint(y), x)
