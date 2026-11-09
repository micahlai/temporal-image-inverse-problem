"""Forward-model physics, statistical checks, and prep2/main interoperability."""

import json
from pathlib import Path
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import pytest
from scipy.signal import convolve2d

from lab8_temporal.inverse_solver import InverseSolver
from lab8_temporal.prep2 import SensorConfig, field_effects, simulate_sensor
from lab8_temporal.vendor.waveprop_numpy import convolve_video
from lab8_temporal.video_io import read_video, write_video


@pytest.mark.parametrize("kernel_shape", [(3, 5), (4, 2), (11, 13)])
def test_adapted_optical_stage_matches_direct_convolution(kernel_shape):
    rng = np.random.default_rng(12)
    video = rng.uniform(size=(7, 9, 3, 5))
    psf = rng.uniform(size=kernel_shape)
    psf /= psf.sum()
    output = convolve_video(video, psf)
    for c in range(video.shape[2]):
        for t in range(video.shape[3]):
            full = convolve2d(video[:, :, c, t], psf)
            y, x = np.array(kernel_shape) // 2
            np.testing.assert_allclose(output[:, :, c, t], full[y:y+7, x:x+9], atol=1e-12)
    np.testing.assert_allclose(output, InverseSolver(psf).A(video), atol=1e-12)


def test_shot_noise_mean_and_signal_dependent_variance():
    optical = np.empty((200, 200, 1, 2))
    optical[..., 0], optical[..., 1] = .1, .6
    cfg = SensorConfig(peak_electrons=2000, read_noise=3, dark_electrons=.2,
                       full_well=10000, bit_depth=16, black_level=1024, prnu=0, dsnu=0)
    measurement, raw, sensitivity, offset, diagnostics = simulate_sensor(optical, cfg, seed=20)
    np.testing.assert_array_equal(sensitivity, 1)
    np.testing.assert_array_equal(offset, 0)
    assert raw.dtype == np.uint16 and raw.max() < 65535
    assert diagnostics["adc_clipped_fraction"] == 0
    gain = diagnostics["gain_dn_per_electron"]
    for t, intensity in enumerate((.1, .6)):
        noise = (measurement - optical)[..., t]
        predicted = ((intensity * cfg.peak_electrons + cfg.dark_electrons + cfg.read_noise**2
                      + 1 / (12 * gain**2)) / cfg.peak_electrons**2)
        assert abs(noise.mean()) < 5 * np.sqrt(predicted / noise.size)
        assert noise.var() == pytest.approx(predicted, rel=.05)
    assert measurement[..., 1].var() > 4 * measurement[..., 0].var()


def test_sensor_fixed_pattern_and_reproducibility():
    optical = np.full((24, 28, 1, 80), .4)
    cfg = SensorConfig(peak_electrons=200000, full_well=1000000, prnu=.05,
                       dsnu=100, dark_electrons=0, read_noise=0)
    first = simulate_sensor(optical, cfg, seed=7)
    repeated = simulate_sensor(optical, cfg, seed=7)
    different = simulate_sensor(optical, cfg, seed=8)
    for a, b in zip(first[:4], repeated[:4]):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], different[0])
    assert not np.array_equal(first[0][..., 0], first[0][..., 1])
    mean = first[0].mean(axis=-1)
    expected = .4 * first[2] + first[3] / cfg.peak_electrons
    assert np.corrcoef(mean.ravel(), expected.ravel())[0, 1] > .99
    assert first[2].shape == optical.shape[:3]  # One fixed map, shared over time.


def test_full_well_saturation_is_nonlinear():
    cfg = SensorConfig(peak_electrons=1000, full_well=100, read_noise=0,
                       dark_electrons=0, prnu=0, dsnu=0)
    optical = np.ones((8, 9, 3, 2))
    a = simulate_sensor(optical, cfg, seed=1)
    b = simulate_sensor(2 * optical, cfg, seed=1)
    np.testing.assert_array_equal(a[1], 4095)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_allclose(a[0], .1)
    assert a[-1]["full_well_saturated_fraction"] == 1


def test_field_effects_static_and_identity():
    video = np.zeros((9, 11, 3, 3))
    video[1:8, 2:9, 1, :] = 1
    np.testing.assert_array_equal(field_effects(video, 0, 0), video)
    modified = field_effects(video, .08, .2)
    assert not np.allclose(video, modified)
    np.testing.assert_array_equal(modified[..., 0], modified[..., 2])
    np.testing.assert_array_equal(modified[:, :, (0, 2)], 0)
    uniform = field_effects(np.ones_like(video), 0, .2)
    np.testing.assert_allclose(uniform[0, 0], .8)
    np.testing.assert_allclose(uniform[4, 5], 1)


@pytest.mark.parametrize("kwargs", [dict(peak_electrons=0), dict(read_noise=-1),
                                    dict(full_well=np.inf), dict(prnu=.5),
                                    dict(bit_depth=7), dict(black_level=4095)])
def test_invalid_sensor_settings(kwargs):
    with pytest.raises(ValueError):
        SensorConfig(**kwargs).validate()


def test_prep2_cli_to_reconstruction(tmp_path):
    video = np.zeros((24, 32, 3, 5))
    video[4:18, 6:22, :, :] = np.array([.3, .6, .8])[None, None, :, None]
    source = tmp_path / "input.mp4"
    psf_path = tmp_path / "psf.tiff"
    write_video(source, video, 12)
    iio.imwrite(psf_path, np.ones((3, 3), dtype=np.float32))
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "prepared2"
    subprocess.run([sys.executable, str(root / "lab8_temporal" / "prep2.py"),
                    str(source), str(psf_path), "--resolution", "17", "13",
                    "--psf-scale", "2", "--noise-seed", "42", "--distortion-k1", "0.03",
                    "--noise-std", "0.001", "--output-dir", str(output_dir)],
                   cwd=root, check=True, capture_output=True, text=True)
    with np.load(output_dir / "dataset.npz") as dataset:
        assert dataset["measurement"].shape == (13, 17, 3, 5)
        assert dataset["raw_codes"].dtype == np.uint16
        metadata = json.loads(str(dataset["metadata"]))
        assert metadata["upstream_commit"] == "84c3e7dd50c7193e4c069a7581740ecbf33a3167"
        assert metadata["sensor"]["noise_std"] == .001
        assert metadata["inverse_model_exact"] is False
        optical = convolve_video(field_effects(dataset["reference"], .03, .1), dataset["psf"])
        np.testing.assert_allclose(dataset["noiseless_measurement"], optical, atol=1e-12)
        cfg = metadata["sensor"]
        gain = metadata["diagnostics"]["gain_dn_per_electron"]
        decoded = (((dataset["raw_codes"].astype(float) - cfg["black_level"]) / gain
                    - cfg["dark_electrons"]) / cfg["peak_electrons"])
        np.testing.assert_array_equal(decoded, dataset["measurement"])
    preview, fps, size = read_video(output_dir / "blurred.mp4")
    assert preview.shape == (13, 17, 3, 5) and fps == 12 and size == (17, 13)
    subprocess.run([sys.executable, "-m", "lab8_temporal.main", str(output_dir / "dataset.npz"),
                    "--num-iters", "2", "--output-dir", str(tmp_path / "result")],
                   cwd=root, check=True, capture_output=True, text=True)
    assert np.load(tmp_path / "result" / "reconstructed.npy").shape == (13, 17, 3, 5)
