"""DiffuserCam-style video simulation with an electron-based CMOS sensor.

Uses an attributed NumPy adaptation of waveprop's PSF convolution stage
(the optical backend used by LenslessPiCam). Sensor and optional field effects
are local extensions, not features calibrated from the supplied TIFF.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

try:
    from .video_io import load_psf, read_video, write_video
    from .vendor.waveprop_numpy import convolve_video, UPSTREAM_COMMIT, UPSTREAM_URL
except ImportError:
    from video_io import load_psf, read_video, write_video
    from vendor.waveprop_numpy import convolve_video, UPSTREAM_COMMIT, UPSTREAM_URL


@dataclass(frozen=True)
class SensorConfig:
    """Illustrative settings, NOT a measured profile of a specific camera.

    Electron quantities refer to one exposure at the output resolution.
    peak_electrons: expected photoelectrons at optical intensity 1.
    read_noise/dsnu: standard deviations in electrons.
    dark_electrons: mean dark charge per pixel per exposure.
    prnu: fractional standard deviation of fixed multiplicative sensitivity.
    noise_std: optional extra Gaussian read noise in normalized scene units.
    """
    peak_electrons: float = 2000.0
    read_noise: float = 3.0
    dark_electrons: float = 0.2
    full_well: float = 10000.0
    bit_depth: int = 12
    black_level: int = 64
    prnu: float = 0.01
    dsnu: float = 0.5
    noise_std: float = 0.0

    def validate(self):
        for name in ("peak_electrons", "read_noise", "dark_electrons", "full_well",
                     "prnu", "dsnu", "noise_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.peak_electrons <= 0 or self.full_well <= 0:
            raise ValueError("peak_electrons and full_well must be positive")
        if not isinstance(self.bit_depth, (int, np.integer)) or not 8 <= self.bit_depth <= 16:
            raise ValueError("bit_depth must be an integer from 8 to 16")
        if not isinstance(self.black_level, (int, np.integer)) or not 0 <= self.black_level < 2**self.bit_depth - 1:
            raise ValueError("black_level must be an integer below the maximum ADC code")
        if self.prnu > 0.25:
            raise ValueError("prnu must be <= 0.25 (fractional standard deviation)")


def field_effects(video, distortion_k1=0.0, vignetting=0.1):
    """Optional static field mismatch applied BEFORE the spatial PSF.

    Output->input map: (u,v) -> (u,v)*(1+k1*r²), r²=1 at the corners.
    Attenuation: 1-vignetting*r². These are controlled phenomenological
    perturbations, not a calibrated field-dependent diffuser PSF model.
    """
    video = np.asarray(video, dtype=float)
    if video.ndim != 4 or not all(video.shape) or not np.isfinite(video).all():
        raise ValueError("Expected a finite, nonempty (H,W,C,T) video")
    if not np.isfinite(distortion_k1) or not -.25 <= distortion_k1 <= .25:
        raise ValueError("distortion_k1 must be between -0.25 and 0.25")
    if not np.isfinite(vignetting) or not 0 <= vignetting < 1:
        raise ValueError("vignetting must be in [0,1)")
    h, w = video.shape[:2]
    yy, xx = np.indices((h, w), dtype=float)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    radius2 = ((yy - cy)**2 + (xx - cx)**2) / max(cy**2 + cx**2, 1)
    illumination = 1 - vignetting * radius2
    if distortion_k1 == 0:
        return video * illumination[:, :, None, None]
    scale = 1 + distortion_k1 * radius2
    coordinates = [cy + (yy - cy) * scale, cx + (xx - cx) * scale]
    result = np.empty_like(video)
    for time in range(video.shape[3]):
        for channel in range(video.shape[2]):
            result[:, :, channel, time] = map_coordinates(
                video[:, :, channel, time], coordinates,
                order=1, mode="constant", cval=0, prefilter=False)
    return result * illumination[:, :, None, None]


def simulate_sensor(optical, config, seed=None):
    """Poisson charge -> full-well clip -> read/offset noise -> ADC.

    PRNU/DSNU maps stay fixed across time; shot/read noise is independent each
    exposure. Returned measurement undoes known ADC gain, black level, and
    mean dark charge to keep its intensity scale usable by the existing FISTA.
    It does not undo fixed-pattern errors, clipping, or digitization.
    """
    config.validate()
    optical = np.asarray(optical, dtype=float)
    if optical.ndim != 4 or not all(optical.shape) or not np.isfinite(optical).all():
        raise ValueError("Expected a finite, nonempty (H,W,C,T) optical video")
    if np.min(optical) < -1e-10:
        raise ValueError("Optical intensities must be nonnegative")
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("noise_seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    fixed_shape = optical.shape[:3]
    # Lognormal sensitivities are positive, with exactly mean 1 and std prnu.
    log_sigma = np.sqrt(np.log1p(config.prnu**2))
    sensitivity = rng.lognormal(-0.5 * log_sigma**2, log_sigma, size=fixed_shape)
    offset = rng.normal(0, config.dsnu, size=fixed_shape)
    expected = (np.maximum(optical, 0) * config.peak_electrons
                * sensitivity[..., None] + config.dark_electrons)
    charge = rng.poisson(expected).astype(float)
    full_well_saturated = charge >= config.full_well
    charge = np.minimum(charge, config.full_well)
    charge += offset[..., None]
    read_std = np.hypot(config.read_noise, config.noise_std * config.peak_electrons)
    charge += rng.normal(0, read_std, size=charge.shape)
    max_code = 2**config.bit_depth - 1
    gain = (max_code - config.black_level) / config.full_well
    codes_float = charge * gain + config.black_level
    adc_clipped = (codes_float < 0) | (codes_float > max_code)
    raw_codes = np.rint(np.clip(codes_float, 0, max_code)).astype(np.uint16)
    measurement = (((raw_codes.astype(float) - config.black_level) / gain
                    - config.dark_electrons) / config.peak_electrons)
    diagnostics = dict(full_well_saturated_fraction=float(full_well_saturated.mean()),
                       adc_clipped_fraction=float(adc_clipped.mean()),
                       sensor_rmse=float(np.sqrt(np.mean((measurement - optical)**2))),
                       gain_dn_per_electron=gain)
    return measurement, raw_codes, sensitivity, offset, diagnostics


def prepare(video_path, psf_path, resolution, output_dir, psf_scale=1.0,
            max_frames=None, noise_std=0.0, noise_seed=None, *,
            peak_electrons=2000.0, read_noise=3.0, dark_electrons=0.2,
            full_well=10000.0, bit_depth=12, black_level=64, prnu=0.01,
            dsnu=0.5, vignetting=0.1, distortion_k1=0.0, input_gamma=1.0):
    """Create a main.py-compatible dataset plus raw ADC and optical truth."""
    config = SensorConfig(peak_electrons, read_noise, dark_electrons, full_well,
                          bit_depth, black_level, prnu, dsnu, noise_std)
    config.validate()
    if noise_seed is not None and (not isinstance(noise_seed, (int, np.integer)) or noise_seed < 0):
        raise ValueError("noise_seed must be a nonnegative integer")
    if not np.isfinite(input_gamma) or input_gamma <= 0:
        raise ValueError("input_gamma must be positive and finite")
    # Validate field controls before decoding video or writing output.
    field_effects(np.zeros((1, 1, 1, 1)), distortion_k1, vignetting)
    if Path(video_path).suffix.lower() != ".mp4":
        raise ValueError("Input video must be an .mp4")
    if Path(psf_path).suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("PSF must be a .tif or .tiff")
    if len(resolution) != 2:
        raise ValueError("resolution must be (width, height)")
    decoded, fps, source_size = read_video(video_path, resolution, max_frames)
    # An optional power-law inverse transfer is an approximation, not raw recovery.
    reference = decoded**input_gamma
    psf = load_psf(psf_path, source_size, resolution, psf_scale)
    object_plane = field_effects(reference, distortion_k1, vignetting)
    optical = np.maximum(convolve_video(object_plane, psf), 0)
    measurement, raw_codes, sensitivity, offset, diagnostics = simulate_sensor(optical, config, noise_seed)
    metadata = dict(
        forward_model="prep2_waveprop_cmos", upstream_url=UPSTREAM_URL,
        upstream_commit=UPSTREAM_COMMIT, upstream_license="MIT",
        implementation="adapted waveprop NumPy convolution stage; local field and CMOS extensions",
        source_video=str(Path(video_path).resolve()), source_psf=str(Path(psf_path).resolve()),
        source_size=list(source_size), target_size=list(resolution), psf_scale=psf_scale,
        fps=fps, frames=reference.shape[3], layout="H,W,C,T", boundary="zero",
        psf_origin="kh//2,kw//2", input_gamma=input_gamma,
        vignetting=vignetting, distortion_k1=distortion_k1,
        noise_seed=None if noise_seed is None else int(noise_seed), sensor=asdict(config),
        diagnostics=diagnostics, calibrated=False,
        inverse_model_exact=False,
        measurement_units="(ADC minus black level)/gain, minus mean dark, divided by peak_electrons",
        inverse_model_note="main.py uses nominal convolution; it does not invert field effects, saturation, or fixed-pattern noise",
        preview="clipped [0,1], no per-frame normalization, no audio; input_gamma undone for display",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "dataset.npz"
    np.savez_compressed(dataset, measurement=measurement, reference=reference,
                        psf=psf, fps=fps, metadata=json.dumps(metadata),
                        noiseless_measurement=optical, raw_codes=raw_codes,
                        sensitivity_map=sensitivity, offset_electrons=offset)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Scientific arrays retain linear intensities; these are display-only previews.
    write_video(output_dir / "blurred.mp4", np.clip(measurement, 0, 1)**(1 / input_gamma), fps)
    write_video(output_dir / "reference.mp4", decoded, fps)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("psf", type=Path)
    parser.add_argument("--resolution", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--psf-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared2"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--noise-seed", type=int, help="Nonnegative seed for all noise and fixed maps")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Extra normalized Gaussian read noise; 0 does NOT disable shot noise")
    parser.add_argument("--peak-electrons", type=float, default=2000,
                        help="Photoelectrons per unit optical intensity (default: 2000); lower is noisier")
    parser.add_argument("--read-noise", type=float, default=3, help="Read noise RMS in electrons (default: 3)")
    parser.add_argument("--dark-electrons", type=float, default=0.2, help="Mean dark charge per exposure (default: 0.2)")
    parser.add_argument("--full-well", type=float, default=10000, help="Saturation charge in electrons (default: 10000)")
    parser.add_argument("--bit-depth", type=int, default=12, help="ADC bits, 8-16 (default: 12)")
    parser.add_argument("--black-level", type=int, default=64, help="ADC black pedestal in digital counts (default: 64)")
    parser.add_argument("--prnu", type=float, default=0.01, help="Fixed sensitivity std as fraction (default: 0.01)")
    parser.add_argument("--dsnu", type=float, default=0.5, help="Fixed readout offset std in electrons (default: 0.5)")
    parser.add_argument("--vignetting", type=float, default=0.1, help="Corner illumination loss, [0,1) (default: 0.1)")
    parser.add_argument("--distortion-k1", type=float, default=0.0, help="Optional radial warp, [-0.25,0.25] (default: 0)")
    parser.add_argument("--input-gamma", type=float, default=1.0, help="Approximate inverse power transfer (default: 1, unchanged)")
    args = vars(parser.parse_args())
    video, psf = args.pop("video"), args.pop("psf")
    try:
        dataset = prepare(video, psf, **args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved sensor simulation: {dataset}")
    print("Defaults are illustrative, not calibrated to your camera.")
    print("main.py can read this dataset; its nominal PSF model does not invert sensor/field effects.")


if __name__ == "__main__":
    main()
