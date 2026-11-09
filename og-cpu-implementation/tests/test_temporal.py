"""Numerical and IO checks for the actual reconstruction model."""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from scipy.signal import convolve2d

from lab8_temporal.inverse_solver import InverseSolver, adjoint_test
from lab8_temporal.prep import prepare
from lab8_temporal.priors import _difference_adjoint, _wavelet, joint_prox, prox_tv, prox_wavelet
from lab8_temporal.video_io import load_psf, read_video, write_video


@pytest.mark.parametrize("shape", [(7, 9, 3, 5), (8, 6, 1, 1)])
@pytest.mark.parametrize("kernel_shape", [(3, 5), (4, 2), (11, 13)])
def test_forward_and_adjoint(shape, kernel_shape):
    rng = np.random.default_rng(7)
    solver = InverseSolver(rng.uniform(size=kernel_shape))
    x = rng.normal(size=shape)
    actual = solver.A(x)
    for channel in range(shape[2]):
        for time in range(shape[3]):
            full = convolve2d(x[:, :, channel, time], solver.h, mode="full")
            h, w = kernel_shape
            expected = full[h // 2:h // 2 + shape[0], w // 2:w // 2 + shape[1]]
            np.testing.assert_allclose(actual[:, :, channel, time], expected, atol=1e-12)
    lhs, rhs = adjoint_test(solver, shape)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_no_forward_time_or_channel_mixing():
    x = np.zeros((9, 9, 3, 5))
    x[4, 4, 1, 2] = 1
    y = InverseSolver(np.ones((3, 3))).A(x)
    assert np.count_nonzero(y[:, :, (0, 2), :]) == 0
    assert np.count_nonzero(y[:, :, :, (0, 1, 3, 4)]) == 0
    np.testing.assert_allclose(y.sum(), 1)


@pytest.mark.parametrize("axes", [(0, 1), (3,)])
@pytest.mark.parametrize("wavelet", ["db4", "haar"])
def test_wavelet_orthonormal_on_odd_shapes(axes, wavelet):
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(7, 9, 3, 5)), rng.normal(size=(7, 9, 3, 5))
    wx = _wavelet(x, axes, wavelet, 4)
    np.testing.assert_allclose(_wavelet(wx, axes, wavelet, 4, True), x, atol=1e-12)
    np.testing.assert_allclose(np.sum(wx**2), np.sum(x**2), rtol=1e-12)
    np.testing.assert_allclose(np.vdot(wx, y), np.vdot(x, _wavelet(y, axes, wavelet, 4, True)), atol=1e-12)


def test_difference_adjoint():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 5, 2, 7))
    for axis in (0, 1, 3):
        dx = np.diff(x, axis=axis)
        p = rng.normal(size=dx.shape)
        np.testing.assert_allclose(np.vdot(dx, p), np.vdot(x, _difference_adjoint(p, axis, x.shape)))


def test_temporal_tv_known_solution_and_no_channel_mixing():
    x = np.zeros((3, 5, 2, 2))
    x[:, :, 0, 1] = 1
    x[:, :, 1, :] = 8
    result = prox_tv(x, 0.2, 1, temporal=True, max_iters=1000, tol=1e-10)
    np.testing.assert_allclose(result[:, :, 0, 0], 0.2, atol=1e-8)
    np.testing.assert_allclose(result[:, :, 0, 1], 0.8, atol=1e-8)
    np.testing.assert_allclose(result[:, :, 1, :], 8, atol=1e-8)
    # Spatial-only TV cannot change a spatially constant flickering video.
    np.testing.assert_allclose(prox_tv(x, 0.2, 1), x)


def test_temporal_wavelet_couples_frames():
    x = np.array([0.0, 1.0]).reshape(1, 1, 1, 2)
    out = prox_wavelet(x, 0.2, 1, temporal=True, wavelet="haar", level=1)
    expected = np.array([0, 1 - np.sqrt(2) * 0.2]).reshape(x.shape)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    # The same value in a one-frame input has no temporal prior.
    np.testing.assert_array_equal(prox_wavelet(x[..., 1:], 0.2, 1, temporal=True), x[..., 1:])


def test_odd_last_frame_participates_in_temporal_wavelets():
    x = np.array([.15, .2, .1, .3, .25]).reshape(1, 1, 1, 5)
    before = prox_wavelet(x, .1, 1, temporal=True)
    x[..., -1] = .8
    after = prox_wavelet(x, .1, 1, temporal=True)
    assert not np.allclose(before[..., :-1], after[..., :-1])


def test_identity_psf_reconstruction_without_penalties():
    x = np.random.default_rng(12).uniform(size=(5, 7, 3, 5))
    solver = InverseSolver(np.ones((1, 1)), num_iters=1,
                           lam_spatial=0, lam_temporal=0)
    np.testing.assert_allclose(solver(x), x, atol=1e-12)


@pytest.mark.parametrize("spatial,temporal", itertools.product(("wavelet", "tv"), repeat=2))
def test_all_combinations_and_independent_weights(spatial, temporal):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(5, 7, 2, 3))
    identity = joint_prox(x, 0, 0, 1, spatial, temporal)
    np.testing.assert_array_equal(identity, x)
    out, info = joint_prox(x, .03, .06, 1, spatial, temporal,
                           max_iters=2000, tol=1e-8, return_info=True)
    assert info["converged"]
    assert out.shape == x.shape and np.isfinite(out).all()
    assert not np.allclose(out, joint_prox(x, 0, .06, 1, spatial, temporal))
    assert not np.allclose(out, joint_prox(x, .03, 0, 1, spatial, temporal))
    # Batch processing is identical to separate channels; time remains joint.
    for c in range(x.shape[2]):
        separate = joint_prox(x[:, :, c:c+1], .03, .06, 1, spatial, temporal,
                              max_iters=2000, tol=1e-10)
        np.testing.assert_allclose(out[:, :, c:c+1], separate, atol=2e-5)


@pytest.mark.parametrize("spatial,temporal", itertools.product(("wavelet", "tv"), repeat=2))
def test_fista_reconstruction_improves_synthetic_mse(spatial, temporal):
    clean = np.zeros((15, 17, 3, 5))
    clean[4:11, 5:12, :, :] = np.array([.4, .7, 1])[None, None, :, None]
    solver = InverseSolver(np.outer([1, 2, 1], [1, 2, 1]), num_iters=50,
                           spatial_prior=spatial, temporal_prior=temporal,
                           lam_spatial=1e-4, lam_temporal=1e-4, prox_iters=500)
    y = solver.A(clean)
    out = solver(y)
    assert out.shape == clean.shape
    assert np.mean((out - clean)**2) < np.mean((y - clean)**2)
    assert solver.losses[-1] < solver.losses[0]
    assert len(solver.prox_diagnostics) == 50


def test_psf_scaling_and_validation(tmp_path):
    psf = np.zeros((9, 9, 3), dtype=np.float32)
    psf[4, 4] = (1, 2, 3)
    path = tmp_path / "psf.tiff"
    iio.imwrite(path, psf, photometric="rgb")
    original = load_psf(path, (100, 80), (100, 80), 1)
    smaller = load_psf(path, (100, 80), (50, 40), 1)
    restored = load_psf(path, (100, 80), (50, 40), 2)
    assert smaller.shape == (5, 5)
    np.testing.assert_array_equal(original, restored)
    np.testing.assert_allclose(smaller.sum(), 1)
    assert np.unravel_index(np.argmax(smaller), smaller.shape) == (2, 2)
    for scale in (0, -1, np.nan):
        with pytest.raises(ValueError):
            load_psf(path, (100, 80), (50, 40), scale)


def test_real_mp4_roundtrip_and_preparation(tmp_path):
    clean = np.zeros((24, 32, 3, 5))
    for t in range(5):
        clean[4:18, 6+t:20+t, :, t] = [.4, .7, .9]
    video = tmp_path / "input.mp4"
    write_video(video, clean, 12)
    psf = tmp_path / "psf.tiff"
    iio.imwrite(psf, np.outer([1, 2, 1], [1, 2, 1]).astype(np.float32))
    dataset_path = prepare(video, psf, (17, 13), tmp_path / "prepared", psf_scale=2)
    with np.load(dataset_path) as dataset:
        assert dataset["reference"].shape == (13, 17, 3, 5)
        assert float(dataset["fps"]) == 12
        np.testing.assert_allclose(InverseSolver(dataset["psf"]).A(dataset["reference"]),
                                   dataset["measurement"], atol=1e-14)
    preview, fps, size = read_video(tmp_path / "prepared" / "blurred.mp4")
    assert preview.shape == (13, 17, 3, 5) and size == (17, 13) and fps == 12
    for bad_size in ((32, 12), (16, 24), (40, 30), (0, 12)):
        with pytest.raises(ValueError, match="smaller"):
            prepare(video, psf, bad_size, tmp_path / "bad")
    root = Path(__file__).resolve().parents[2]
    # Exercise the noise CLI and verify seeded preparation matches the API.
    subprocess.run([sys.executable, "-m", "lab8_temporal.prep", str(video), str(psf),
                    "--resolution", "17", "13", "--psf-scale", "2",
                    "--noise-std", "0.03", "--noise-seed", "42",
                    "--output-dir", str(tmp_path / "noisy")],
                   cwd=root, check=True, capture_output=True, text=True)
    repeated = prepare(video, psf, (17, 13), tmp_path / "repeated", psf_scale=2,
                       noise_std=.03, noise_seed=42)
    with np.load(tmp_path / "noisy" / "dataset.npz") as noisy, np.load(dataset_path) as clean_data, np.load(repeated) as again:
        np.testing.assert_array_equal(noisy["measurement"], again["measurement"])
        np.testing.assert_array_equal(noisy["reference"], clean_data["reference"])
        np.testing.assert_array_equal(noisy["psf"], clean_data["psf"])
        noise = noisy["measurement"] - clean_data["measurement"]
        assert abs(noise.mean()) < .003
        assert abs(noise.std() - .03) < .003
        assert noisy["measurement"].min() < 0  # Scientific data is not clipped.
        metadata = json.loads(str(noisy["metadata"]))
        assert metadata["noise_std"] == .03 and metadata["noise_seed"] == 42
    for bad_std in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="noise_std"):
            prepare(video, psf, (17, 13), tmp_path / "bad", noise_std=bad_std)
    with pytest.raises(ValueError, match="noise_seed"):
        prepare(video, psf, (17, 13), tmp_path / "bad", noise_seed=-1)
    subprocess.run([sys.executable, "-m", "lab8_temporal.main", str(dataset_path),
                    "--lambda-spatial", "-4", "--lambda-temporal", "-6",
                    "--num-iters", "3", "--output-dir", str(tmp_path / "result")],
                   cwd=root, check=True, capture_output=True, text=True)
    result = np.load(tmp_path / "result" / "reconstructed.npy")
    assert result.shape == (13, 17, 3, 5)
    metrics = json.loads((tmp_path / "result" / "metrics.json").read_text())
    assert "reconstructed_mse" in metrics
    assert metrics["lambda_spatial"] == 1e-4
    assert metrics["lambda_temporal"] == 1e-6
    subprocess.run([sys.executable, str(root / "lab8_temporal" / "main.py"), str(video),
                    "--psf", str(psf), "--num-iters", "2", "--output-dir", str(tmp_path / "real")],
                   cwd=root, check=True, capture_output=True, text=True)
    assert np.load(tmp_path / "real" / "reconstructed.npy").shape == clean.shape
    default_metrics = json.loads((tmp_path / "real" / "metrics.json").read_text())
    assert default_metrics["lambda_spatial"] == 1e-5
    assert default_metrics["lambda_temporal"] == 1e-5
