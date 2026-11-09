# Third-party provenance

`waveprop_numpy.py` adapts the NumPy PSF-convolution portion of
`FarFieldSimulator.set_psf` / `FarFieldSimulator.propagate` from:

- Repository: https://github.com/ebezzam/waveprop
- File: `waveprop/simulation.py`
- Commit: `84c3e7dd50c7193e4c069a7581740ecbf33a3167`
- License: MIT; complete text in `WAVEPROP_LICENSE.txt`
- Copyright: 2022 Eric Bezzam

Changes: standalone 4D helper; independent kernel and image sizes; integer-center
kernel origin including even kernels; float64 output; no geometry preprocessing,
SNR rescaling, per-image normalization, or built-in quantization. Local sensor
and field extensions live in `prep2.py`. The full upstream simulator is not
bundled, and no LenslessPiCam source code is copied.
