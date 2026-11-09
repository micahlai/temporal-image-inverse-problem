# prep2: DiffuserCam-style optics and a CMOS sensor

`prep2.py` preserves `prep.py`'s required arguments (`video`, TIFF `psf`, and
`--resolution WIDTH HEIGHT`), PSF scale convention, frame limits, and output
layout. It adds an electron-based sensor simulation with noise enabled by
default. No additional dependencies are needed beyond `requirements.txt`.

```sh
python lab8_temporal/prep2.py input.mp4 psf.tiff \
  --resolution 160 120 --psf-scale 1 --max-frames 30 \
  --noise-seed 42 --output-dir lab8_temporal/prepared2

python lab8_temporal/main.py lab8_temporal/prepared2/dataset.npz \
  --spatial-prior wavelet --temporal-prior tv \
  --lambda-spatial -5 --lambda-temporal -5 \
  --output-dir lab8_temporal/results/prep2
```

Both output dimensions must be smaller than the original video. As in `prep.py`,
`psf_scale=1` means a PSF pixel corresponds to one original-video pixel; resizing
also scales the PSF. One normalized 2D kernel is shared across channels/frames.

## Open-source basis and scope

The selected repository is [waveprop](https://github.com/ebezzam/waveprop), whose
`FarFieldSimulator` is the optical backend wrapped by
[LenslessPiCam](https://lensless.readthedocs.io/en/stable/_modules/lensless/utils/simulation.html).
Its [simulation pipeline](https://github.com/LCAV/LenslessPiCam/blob/main/docs/source/simulation.rst)
uses PSF convolution, optional noise, and quantization.

This implementation vendors **only an adaptation of waveprop's NumPy spatial
convolution stage**, pinned to commit
[`84c3e7dd50c7193e4c069a7581740ecbf33a3167`](https://github.com/ebezzam/waveprop/blob/84c3e7dd50c7193e4c069a7581740ecbf33a3167/waveprop/simulation.py).
See `vendor/waveprop_numpy.py` and its accompanying MIT license. It does not
import or claim to run the complete upstream `FarFieldSimulator`. This keeps the
existing NumPy/SciPy environment sufficient, without adding Torch/torchvision.

Local changes preserve arbitrary image/kernel sizes, odd dimensions, 4D layout,
and this lab's PSF origin convention. Physical scene-distance/object-height
resizing is not used: the supplied video already defines the reconstruction
plane. The optical calculation is checked against direct spatial convolution.
Upstream's SNR-based noise and per-image maximum normalization/8-bit conversion
are replaced by the local fixed-exposure electron/ADC model below. The field
perturbations are also local extensions, not upstream calibrated diffuser effects.

For incoherent imaging with a single calibrated PSF, the optical model remains
linear. See the [original DiffuserCam paper](https://doi.org/10.1364/optica.5.000001).
Here the nonlinearities are **full-well saturation and ADC clipping/quantization**.
The optional radial warp is nonlinear in spatial coordinates but remains linear
in scene intensity. A single TIFF cannot determine field-dependent PSFs,
depth effects, coherent interference, or measured geometric distortion.

## Forward pipeline

1. Resize the video and PSF using the same conventions as `prep.py`.
2. Optionally approximate inverse gamma: `reference = decoded_video**input_gamma`.
3. Apply optional static radial coordinate distortion and illumination falloff.
4. Convolve spatially using the adapted waveprop stage; no temporal blur.
5. Convert to expected electrons with a fixed exposure scale and fixed pixel
   sensitivity map; add expected dark charge.
6. Draw Poisson electrons, with variance equal to the expected count.
7. Clip charge at the full-well capacity, then add fixed pixel readout offsets
   and independent zero-mean Gaussian read noise.
8. Apply ADC gain/black pedestal, clip, and quantize to integer digital counts.
9. Undo known gain, black pedestal, and *mean* dark charge to supply `main.py`
   with a floating-point measurement on the reference's intensity scale.

The sensor ingredients follow the physical model described by
[EMVA 1288 Linear](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf).
This is not an EMVA-certified simulator or a measured camera profile. No
per-frame exposure adjustment or peak normalization is performed.

## Controls

| Flag | Default | Meaning |
| --- | --- | --- |
| `--peak-electrons` | `2000` | Expected photoelectrons at optical intensity 1; lower values increase relative shot noise |
| `--read-noise` | `3` | Read-noise standard deviation in electrons per exposure |
| `--dark-electrons` | `0.2` | Mean dark electrons per pixel per exposure; also contributes shot noise |
| `--full-well` | `10000` | Charge saturation threshold in electrons |
| `--bit-depth` | `12` | ADC bit depth, from 8 through 16 |
| `--black-level` | `64` | ADC black pedestal in digital counts |
| `--prnu` | `0.01` | Fractional standard deviation of fixed pixel sensitivity (1%) |
| `--dsnu` | `0.5` | Standard deviation of fixed pixel readout offsets, in electrons |
| `--noise-std` | `0` | Extra Gaussian read noise on the normalized scene scale; combined in quadrature with electron read noise |
| `--noise-seed` | random | Seed for shot/read noise and fixed-pattern maps |
| `--vignetting` | `0.1` | Illustrative 10% corner illumination loss; set 0 to disable |
| `--distortion-k1` | `0` | Optional radial coordinate warp, allowed range -0.25 to 0.25 |
| `--input-gamma` | `1` | Input power-law inverse transfer; default preserves prep.py's intensity convention |

These are **illustrative** settings. The 3-electron read noise, 10,000-electron
well, and 12-bit ADC are representative of a class of CMOS sensors, not of your
specific camera: for example, this
[manufacturer report](https://softwareservices.flir.com/BFS-PGE-123S6P/latest/EMVA/EMVA.html)
lists 2.68 electrons, 10,995 electrons, and 12 bits for an IMX253-based device.
Exposure, dark charge, fixed-pattern variation, and vignetting are adjustable
experiment assumptions. Actual noise depends on exposure, illumination, sensor,
and temperature, which cannot be inferred from an MP4 and PSF alone.

`--noise-std 0` disables only the *extra* Gaussian term. Photon/read noise and
sensor digitization remain active. For stronger low-light degradation, try
`--peak-electrons 200`; to exercise saturation, increase `--peak-electrons`
above `--full-well` for bright scenes. Saturation is enabled by default, but
ordinary inputs at the default exposure may never reach it.

To experiment with geometric mismatch, add `--distortion-k1 0.03`. Coordinates
are mapped output-to-input by `(u,v) -> (u,v)*(1+k1*r²)`, with `r²=1` at image
corners. Off-image samples are zero. Illumination is multiplied by
`1-vignetting*r²` before convolution. These are explicit phenomenological
perturbations; they are not inferred from the PSF or guaranteed to match a diffuser.

Decoded MP4s are generally display-encoded, not measured linear radiance.
`--input-gamma 2.2` is an optional approximation; it does not recover an unknown
camera ISP or precisely decode every transfer function. Reference/measurement
arrays are then in the resulting linearized scale. prep2's MP4 previews undo
that power for display; main.py's current reconstruction preview does not.

## Saved data and reconstruction limits

`dataset.npz` contains the same `measurement`, `reference`, `psf`, `fps`, and
JSON `metadata` keys expected by `main.py`, plus:

- `noiseless_measurement`: after optional field effects and convolution, before
  fixed-pattern response and sensor noise.
- `raw_codes`: unsigned 16-bit storage containing the selected ADC's integer
  values; a 12-bit run has codes from 0 to 4095.
- `sensitivity_map`: one `(H,W,C)` fixed multiplicative map used for every frame.
- `offset_electrons`: one `(H,W,C)` fixed readout-offset map used for every frame.

`metadata.json` records all parameters, the upstream commit, ADC gain, sensor
RMSE, full-well saturation fraction, and ADC clipping fraction. `blurred.mp4`
and `reference.mp4` are lossy previews at the requested resolution and original
FPS, without audio. Floating-point measurements may be below 0 or above 1 after
known electronic offsets/gain are removed; the saved arrays are not clipped.

The existing FISTA solver uses the stored nominal PSF. It does **not** model
vignetting, the radial warp, fixed-pattern variations, saturation, or ADC
quantization in its data term. Thus these datasets intentionally test model
mismatch; perfect recovery is not expected. To isolate sensor noise, set
`--vignetting 0 --distortion-k1 0 --prnu 0 --dsnu 0` and use an unsaturated
exposure. The reference remains the unwarped scene, so reported MSE includes
geometric error when distortion is enabled.

This is an RGB-plane surrogate, not a Bayer/CFA and demosaicing simulator.
It does not infer a 3D scene, simulate rolling shutter or motion within exposure,
or rescale well capacity/noise with output resolution. Treat the output grid as
the sensor grid and supply parameters appropriate to that grid/exposure. All
frames are loaded into memory, as in the rest of this lab.
