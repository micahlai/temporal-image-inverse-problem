# PyTorch joint video deconvolution

A GPU port of `../og-cpu-implementation`. The original implementation and its
datasets are unchanged. Reconstruction still optimizes the **entire (H,W,C,T)
video jointly**, using the same FISTA and joint spatial/temporal proximal method.

## Conda setup

From this directory:

```sh
conda env create -f environment.yml
conda activate temporal-torch
```

Or create the environment manually, from the repository root:

```sh
conda create -n temporal-torch python=3.11 pip -y
conda activate temporal-torch
python -m pip install -r torch-implementation/requirements.txt
```

The requirements use PyTorch 2.15 or newer, tested with native FFTs on Apple MPS.
For NVIDIA, install a PyTorch build compatible with your CUDA driver using the
[official installer](https://pytorch.org/get-started/locally/), then install the
remaining requirements. The program reports the device and precision used.

## Run existing datasets

From the repository root, with the Conda environment active:

```sh
python torch-implementation/main.py og-cpu-implementation/data/pure_conv/dataset.npz \
  --device auto --spatial-prior wavelet --temporal-prior tv \
  --lambda-spatial -5 --lambda-temporal -5 --num-iters 200 \
  --output-dir torch-implementation/results/wavelet_tv
```

To reproduce the original `run.sh` comparison (four combinations, weights
`1e-7`, 250 iterations), this launcher activates the environment via `conda run`:

```sh
bash torch-implementation/run.sh --show-plot false
```

Additional flags passed to `run.sh` override its defaults. For example,
`--num-iters 10 --output-dir torch-implementation/results/short` runs a short
comparison in a separate output directory.

- `--device auto`: CUDA first, then Apple MPS, then CPU. Explicit `--device mps`,
  `--device cuda:0`, or `--device cpu` is also supported. An unavailable explicit
  device raises an error; it does not silently fall back.
- `--dtype auto`: float32 on GPU, float64 on CPU. Use `--device cpu --dtype float64`
  for close numerical comparison with NumPy. CUDA also supports float64.
  MPS requires float32. Precision can change rounding and the iteration at which
  the inner gap tolerance is reached; float32 results are not bit-identical.
- All existing reconstruction flags are retained, including `--all-priors true`,
  `--show-plot false`, `--prox-iters`, `--prox-tol`, `--wavelet`, and
  `--wavelet-level`. CLI lambdas remain **base-10 exponents**; `-5` means `1e-5`.
- `--all-priors true` runs wavelet/wavelet, wavelet/TV, TV/wavelet and TV/TV
  sequentially, saving the same log-loss versus linear-iteration comparison.

Both original preparation scripts' `.npz` datasets work directly. Keep using
`og-cpu-implementation/prep.py` or `prep2.py` to generate data with the original
resizing, PSF scaling, noise, and sensor-model conventions. No data conversion or
regeneration is necessary. The exact stored PSF is used without rescaling.
As before, the inverse model does not undo `prep2.py`'s nonlinear sensor effects.

Real videos use the same interface:

```sh
python torch-implementation/main.py blurry.mp4 --psf psf.tiff --psf-scale 1 \
  --device auto --output-dir torch-implementation/results/real
```

Outputs remain `reconstructed.npy` (unclipped), `reconstructed.mp4` (clipped
preview), `data_losses.npy`, and `metrics.json`. Metrics additionally record the
device, precision, and PyTorch version. All-priors runs save each combination
in its own subdirectory and write `loss_all_priors.png`.

With `--all-priors true`, the output directory also contains:

- `loss_all_priors.csv`: one row per iteration per combination, with spatial
  and temporal priors, iteration, data loss, inner proximal iteration count,
  primal-dual gap, and convergence flag. These are the original loss values,
  including zero, before the log plot masks them.
- `metrics_all_priors.csv`: one row per combination, including all fields from
  its `metrics.json`: settings, shape, device, final loss, convergence counts,
  ground-truth source, and blurred/reconstructed MSE and PSNR in dB.

### Ground truth and PSNR

The dataset's `reference` array is used automatically when present (including
`og-cpu-implementation/data/pure_conv/dataset.npz`). To override it or evaluate
a real video, add `--gt "path"`:

```sh
python torch-implementation/main.py og-cpu-implementation/data/pure_conv/dataset.npz \
  --all-priors true --gt "path/to/ground_truth.npy" --show-plot false \
  --output-dir torch-implementation/results/comparison
```

Ground truth can be a `.npy` array in `(H,W,C,T)` layout, an `.npz` containing
`reference`, or an RGB `.mp4`. Arrays must match the measurement shape and use
normalized `[0,1]` values. Higher-resolution MP4s are spatially downsampled with
the same Lanczos resizing as preparation. Smaller MP4s are rejected. Frame
count and FPS must match; no temporal trimming, resampling, or alignment is
performed. If preparation used only the first N frames, use its stored
`reference` or an explicitly trimmed ground-truth video. MP4 compression can
affect the score; the stored floating-point reference is preferable.

PSNR is `10*log10(1/MSE)`, using one MSE pooled over all pixels, channels, and
frames of the **unclipped** result. It is not an average of per-frame PSNR.
Both blurred and reconstructed PSNR are printed and recorded in `metrics.json`,
including single-prior runs. An exact match has `"Infinity"` PSNR in JSON/CSV.
Without ground truth, MSE/PSNR are `null` in JSON and empty in CSV. CSV export
covers histories and metrics; full reconstruction arrays remain in `.npy`.

## Preserved method

The objective is unchanged:

```text
0.5 * ||A(x) - measurement||²
  + lam_spatial * R_spatial(x) + lam_temporal * R_temporal(x)
```

- The shared PSF is nonnegative and normalized by its sum, centered at
  `(kh//2, kw//2)`. FFTs act only on axes 0 and 1, using the original padded
  sizes, zero boundaries and same-size crop. The adjoint embeds that crop
  before correlation.
- FISTA starts from zero, uses the same step (default 1), momentum recurrence,
  and fixed outer iteration count. ISTA and gradient descent remain available
  through the Python API.
- TV is anisotropic L1 of open-boundary differences on axes `(0,1)` or `(3,)`.
  Wavelets use the same orthogonal filters (`db4`, level 4 by default),
  periodization, coefficient ordering, and odd-prefix/tail convention. All
  coefficients, including coarse coefficients, are penalized. Channels stay
  independent. A single time sample disables the temporal penalty.
- The sum of both penalties is solved with the same accelerated projected dual
  iterations, operator-norm bounds, zero initialization, relative primal-dual
  gap check on every iteration, tolerance, and iteration cap. There is no
  frame splitting, motion compensation, new positivity constraint, or warm start.
- FFTs, wavelet filtering/synthesis, TV differences, dual variables, and FISTA
  updates stay on the selected device. PyWavelets supplies only constant filter
  coefficients. NumPy handles file I/O and reports; scalar losses and gap checks
  synchronize with the host. Autograd is disabled for reconstruction.

The full video plus working buffers must fit in device memory. GPU acceleration
does not guarantee a speedup for tiny inputs or many short transforms. The same
inner-solve accuracy controls still apply; unconverged solves produce the same
warning and diagnostics as the CPU implementation.

## Python API

Because the requested directory name contains a hyphen, add it to your import
path (or run a script from this directory):

```python
import sys
import numpy as np
sys.path.insert(0, "torch-implementation")
from inverse_solver import InverseSolver

with np.load("og-cpu-implementation/data/pure_conv/dataset.npz") as data:
    solver = InverseSolver(data["psf"], device="auto", num_iters=200,
                           spatial_prior="wavelet", temporal_prior="tv",
                           lam_spatial=1e-5, lam_temporal=1e-5)
    result = solver(data["measurement"])  # Tensor on solver.device, (H,W,C,T).
    array = result.cpu().numpy()           # Transfer once, for output.
```

The Python API accepts absolute lambda weights, including zero. It accepts
NumPy arrays and tensors and returns tensors. Standalone prior functions
preserve tensor device/precision; NumPy inputs to priors use CPU float64.

## Verify

```sh
conda run --no-capture-output -n temporal-torch python -m pytest torch-implementation/tests -q
```

Tests compare the port directly against `og-cpu-implementation`: convolution
and its adjoint, even/odd/oversized PSFs, odd/even wavelet transforms, all four
priors, independent weights, complete FISTA/ISTA/gradient-descent trajectories,
stopping diagnostics, temporal boundaries, channel isolation, reconstruction
improvement, MP4 input/output, and all-priors plotting. CUDA and MPS checks run
when that hardware is available and otherwise report skips.
