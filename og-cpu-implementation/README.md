# Lab 8: joint video deconvolution

This is the video version of `lab8`; the original lab is unchanged. Reconstruction
optimizes the entire video at once with FISTA. The forward model convolves only
spatially, with the same PSF for every frame and color channel. The proximal
update couples frames through a separately weighted temporal prior.

Arrays use **(height, width, channel, time)**, extending Lab 8's NumPy image layout
with time last. CLI resolution arguments use **width height**. If your array is
literally `(X,Y,C,T)`, transpose its first two axes before using this interface.
All frames are loaded into memory; begin with a short, low-resolution clip.

## Install

From the repository root, using your Python environment:

```sh
python -m pip install -r lab8_temporal/requirements.txt
```

## Prepare a synthetic experiment

For a sensor simulation with shot/read noise, fixed-pattern variation,
saturation, quantization, and optional geometric distortion, use
[`prep2.py`](PREP2.md). It accepts the same video, PSF, resolution, and PSF-scale
arguments and produces a dataset readable by `main.py`.

```sh
python lab8_temporal/prep.py input.mp4 psf.tiff \
  --resolution 160 120 --psf-scale 1 \
  --noise-std 0.01 --noise-seed 42 \
  --max-frames 30 --output-dir lab8_temporal/prepared
```

Both target width and target height must be strictly smaller than the input.
`--max-frames` is optional; omit it to use all frames. Frame rate is preserved.
Resizing is performed before convolution. Arbitrary aspect ratios are allowed;
when those ratios differ, the PSF scales differently along its two spatial axes.

`--psf-scale 1` means a TIFF pixel has the size of an original input-video pixel.
The effective horizontal/vertical scales are:

```text
scale_x = psf_scale * target_width  / original_width
scale_y = psf_scale * target_height / original_height
```

For example, resizing 640×480 to 160×120 at scale 1 reduces the PSF footprint
by four. Scale 2 doubles that footprint relative to scale 1. PSF resampling
anti-aliases shrinking kernels and keeps the optical origin at the center.
A TIFF may contain a grayscale or RGB(A) image; RGB channels are averaged into
one shared kernel, ignoring alpha. The PSF must be nonnegative and nonzero and
is normalized to unit sum. It is assumed centered at `(kh//2, kw//2)`.

`--noise-std` controls additive zero-mean Gaussian noise **after convolution**.
It is an absolute standard deviation on the [0,1] intensity scale, not an
exponent: `0.01` is about 2.55 intensity levels in an 8-bit image. Noise is
independent across pixels, channels, and frames. The default is `0` (no noise).
Use `--noise-seed 42` (or any nonnegative integer) to reproduce the same noise
for the same inputs; omitting the seed generates fresh noise each run.

Preparation writes:

- `dataset.npz`: floating-point `measurement`, `reference`, the exact processed
  `psf`, scalar `fps`, and JSON `metadata`. `measurement` includes any added
  noise; `reference` remains clean. All video arrays are `(H,W,C,T)`.
- `blurred.mp4` and `reference.mp4`: viewable previews.
- `metadata.json`: source paths, dimensions, scale, FPS, noise standard
  deviation/seed, and model conventions.

Use `dataset.npz` for consistent synthetic inversion. The preview MP4s use lossy
H.264, clip to [0,1], quantize to 8-bit, and omit audio. They do not rescale
individual frames or change the target dimensions, including odd dimensions.
The arrays retain their original values, including noisy measurements below
zero or above one. `main.py` automatically reconstructs the saved noisy
measurement using the stored PSF; it needs no extra noise parameter.

## Reconstruct

```sh
python lab8_temporal/main.py lab8_temporal/prepared/dataset.npz \
  --spatial-prior wavelet --temporal-prior tv \
  --lambda-spatial -5 --lambda-temporal -5 \
  --num-iters 200 --output-dir lab8_temporal/results/wavelet_tv
```

Choose **exactly one** prior for each axis group:

| Spatial | Temporal |
| --- | --- |
| wavelet | wavelet |
| wavelet | tv |
| tv | wavelet |
| tv | tv |

The CLI lambda arguments are **base-10 exponents**: `--lambda-spatial -5`
means a spatial weight of `1e-5`. Both default to `-5` (weights of `1e-5`).
For example, `-3` means `0.001`; `0` means `1`, not a disabled penalty.
The Python API still accepts absolute weights, including zero to disable a
penalty. `metrics.json` records the converted absolute weights. Weights assume
inputs on the [0,1] intensity scale; defaults are starting points, not tuned
values. The synthetic input automatically supplies its stored PSF and
FPS. Passing another PSF or a scale override with a dataset is rejected.

Outputs are `reconstructed.npy` (unclipped float array), `reconstructed.mp4`
(clipped preview), `data_losses.npy`, and `metrics.json`. Synthetic runs report
blurred and reconstructed MSE against the resized reference, before display
clipping. The loss trace is only the data-fidelity term; it is not the complete
regularized objective and need not decrease on every FISTA step.

To compare all four combinations with the same lambdas and iteration count:

```sh
python lab8_temporal/main.py lab8_temporal/prepared/dataset.npz \
  --all-priors true --lambda-spatial -5 --lambda-temporal -5 \
  --num-iters 200 --output-dir lab8_temporal/results/all_priors
```

`--all-priors true` ignores the selected `--spatial-prior` and `--temporal-prior`
and runs wavelet/wavelet, wavelet/TV, TV/wavelet, and TV/TV sequentially. Each run
starts from zero and gets its own `spatial_<name>_temporal_<name>` output folder.
Matplotlib displays the four data-loss curves together, using **logarithmic
loss on the y-axis** and **linear iteration on the x-axis**, starting at iteration
1. It also saves `loss_all_priors.png` in the main output folder. Nonpositive
losses are masked in the log plot; saved `data_losses.npy` arrays are unchanged.

For a noninteractive run, add `--show-plot false`; the PNG is still saved.
`--all-priors false` (the default) preserves the original single-combination
behavior and output layout. Each of the four runs uses the full video jointly.

For a real captured MP4:

```sh
python lab8_temporal/main.py blurry.mp4 --psf psf.tiff --psf-scale 1 \
  --spatial-prior tv --temporal-prior wavelet \
  --lambda-spatial -5 --lambda-temporal -5 \
  --output-dir lab8_temporal/results/real
```

Real inputs are not resized. Their scale parameter is relative to pixels of the
supplied blurry video. Kernel calibration, noise, video compression, and boundary
mismatch can affect results; real captures have no reference MSE.

## Solver and prior details

```python
import numpy as np
from lab8_temporal.inverse_solver import InverseSolver

with np.load("lab8_temporal/prepared/dataset.npz") as data:
    solver = InverseSolver(
        data["psf"], algorithm="fista", num_iters=200,
        spatial_prior="wavelet", temporal_prior="tv",
        lam_spatial=1e-5, lam_temporal=1e-5,
    )
    reconstructed = solver(data["measurement"])
```

The objective is:

```text
0.5 * ||A(x) - measurement||²
    + lam_spatial  * R_spatial(x)
    + lam_temporal * R_temporal(x)
```

- `A` is linear spatial convolution with zero boundaries and same-size cropping.
  `A_adjoint` is its exact transpose, including the crop. FFTs use only axes 0
  and 1. Kernel normalization is by sum, rather than the original lab's norm.
- Padding is internal to the forward operator. The optimization variable and
  returned result have the measurement's shape, with no extra spatial or time
  samples. A conservative step size of 1 is valid for the unit-sum nonnegative
  PSF; custom steps must lie in (0,1]. ISTA and unregularized gradient descent
  are also available through the Python API.
- **TV:** anisotropic L1 differences along axes 0 and 1 for space, or adjacent
  frames on axis 3 for time. There is no last-to-first TV connection. `prox_tv`
  uses an iterative dual solve instead of Lab 8's approximate Haar average.
- **Wavelets:** orthonormal, separable multilevel transforms along spatial axes
  or time, with `db4`, level 4 by default. `--wavelet` and `--wavelet-level`
  configure both groups. All coefficients, including coarse coefficients,
  receive an L1 penalty, matching Lab 8's sparsity convention. This can shrink
  constant signals. Periodization can connect the ends of the time interval.
- Odd lengths use a square orthonormal transform: each level transforms its
  even prefix and carries the unmatched last sample into the next coarse
  level. Levels stop when a single sample remains. This is an explicit boundary
  convention, not PyWavelets' default odd-length extension. A one-frame video
  disables its temporal penalty entirely.
- Neither prior mixes color channels. Temporal regularization assumes the same
  pixel coordinates across frames; there is no optical-flow motion compensation.
- A single joint proximal solve handles the **sum** of the two penalties. It
  uses accelerated projected-gradient iterations on the dual and checks a
  relative primal-dual gap. This is an approximate inner solve bounded by
  `--prox-iters` (default 200) and `--prox-tol` (default 1e-5). The solver warns
  when the cap is reached, records `prox_diagnostics`, and saves the count in
  `metrics.json`. Increase the cap or adjust tolerance when needed. Finite inner
  tolerances mean this is inexact FISTA, not an exact-prox convergence guarantee.
- No nonnegativity or [0,1] constraint is silently included in optimization.
  Only the exported preview is clipped. Several full-video working arrays and
  FFT buffers are required, so memory use exceeds the input array size.

PyWavelets' transform conventions are described in its
[DWT documentation](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html).

## Verify

```sh
python -m pytest lab8_temporal/tests -q
```

The checks cover convolution against direct spatial convolution, the adjoint,
all four prior combinations, independent lambdas, temporal coupling, channel
isolation, odd sizes, singleton time, wavelet orthonormality, PSF scaling,
synthetic reconstruction improvement, and real MP4 preparation/reconstruction.
