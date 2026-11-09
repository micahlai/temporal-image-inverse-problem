"""NumPy PSF-convolution stage adapted from waveprop FarFieldSimulator.

Copyright (c) 2022 Eric Bezzam. MIT license: WAVEPROP_LICENSE.txt.
Upstream: https://github.com/ebezzam/waveprop/blob/
84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/simulation.py

Local changes: accept independent image/kernel sizes, process H,W,C,T arrays,
use origin kh//2,kw//2 (including even kernels), preserve float64 radiometry.
Object geometry/resizing is handled by prep2; shot noise and ADC are handled
by its electron-based sensor model. This is NOT the full upstream simulator.
"""

import numpy as np

UPSTREAM_COMMIT = "84c3e7dd50c7193e4c069a7581740ecbf33a3167"
UPSTREAM_URL = f"https://github.com/ebezzam/waveprop/blob/{UPSTREAM_COMMIT}/waveprop/simulation.py"


def convolve_video(video, psf):
    """Zero-boundary linear spatial convolution, independently per C,T."""
    video, psf = np.asarray(video, dtype=float), np.asarray(psf, dtype=float)
    if video.ndim != 4 or psf.ndim != 2 or not all(video.shape) or not all(psf.shape):
        raise ValueError("Expected (H,W,C,T) video and 2D PSF")
    if not np.isfinite(video).all() or not np.isfinite(psf).all():
        raise ValueError("Video and PSF must be finite")
    conv_dim = np.asarray(video.shape[:2])
    fft_shape = conv_dim + np.asarray(psf.shape) - 1
    axes = (0, 1)
    H = np.fft.rfft2(psf, s=fft_shape, axes=axes)[:, :, None, None]
    I = np.fft.rfft2(video, s=fft_shape, axes=axes)
    image_plane = np.fft.irfft2(H * I, s=fft_shape, axes=axes)
    y_pad_edge, x_pad_edge = np.asarray(psf.shape) // 2
    return image_plane[y_pad_edge:y_pad_edge + conv_dim[0],
                       x_pad_edge:x_pad_edge + conv_dim[1]].copy()
