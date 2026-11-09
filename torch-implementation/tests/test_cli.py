"""CLI comparison behavior and Matplotlib axis/curve verification."""

import itertools
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from torch_impl.main import plot_prior_losses, load_ground_truth, quality_metrics
from torch_impl.video_io import write_video, read_video
import imageio.v3 as iio
import torch


@pytest.mark.parametrize("losses", [[4.0, 1.0, .1], [1.0, 0.0, .1], [0.0, 0.0, 0.0]])
def test_comparison_plot_axes_and_four_curves(losses):
    pairs = list(itertools.product(("wavelet", "tv"), repeat=2))
    figure = plot_prior_losses({pair: losses for pair in pairs})
    axis = figure.axes[0]
    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "log"
    assert len(axis.lines) == 4
    assert len(axis.get_legend().get_texts()) == 4
    for pair, line in zip(pairs, axis.lines):
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
        assert pair[0] in line.get_label() and pair[1] in line.get_label()
        np.testing.assert_array_equal(np.ma.getmaskarray(line.get_ydata()), np.asarray(losses) <= 0)
    figure.canvas.draw()
    plt.close(figure)


def test_cli_all_priors_and_explicit_false(tmp_path):
    root = Path(__file__).resolve().parents[2]
    dataset = tmp_path / "dataset.npz"
    measurement = np.random.default_rng(31).uniform(size=(9, 11, 3, 5))
    np.savez(dataset, measurement=measurement, reference=measurement,
             psf=np.ones((1, 1)), fps=12)
    gt = tmp_path / "ground_truth.npy"
    reference = .8 * measurement
    np.save(gt, reference)
    environment = dict(os.environ, MPLBACKEND="Agg")
    base = [sys.executable, str(root / "torch-implementation/main.py"), str(dataset),
            "--device", "cpu",
            "--spatial-prior", "tv", "--temporal-prior", "tv",
            "--lambda-spatial", "-2", "--lambda-temporal", "-3",
            "--num-iters", "4", "--show-plot", "false"]
    comparison = tmp_path / "all"
    subprocess.run(base + ["--all-priors", "true", "--gt", str(gt), "--output-dir", str(comparison)],
                   cwd=root, env=environment, check=True, capture_output=True, text=True)
    with (comparison / "loss_all_priors.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    with (comparison / "metrics_all_priors.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert len(history) == 4 * 4 and len(summaries) == 4
    folders = list(comparison.glob("spatial_*_temporal_*"))
    assert len(folders) == 4
    for spatial, temporal in itertools.product(("wavelet", "tv"), repeat=2):
        folder = comparison / f"spatial_{spatial}_temporal_{temporal}"
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["spatial_prior"] == spatial
        assert metrics["temporal_prior"] == temporal
        assert metrics["lambda_spatial"] == 1e-2
        assert metrics["lambda_temporal"] == 1e-3
        assert metrics["ground_truth"] == str(gt.resolve())
        reconstructed = np.load(folder / "reconstructed.npy")
        expected_mse = np.mean((reconstructed - reference)**2)
        assert metrics["reconstructed_mse"] == pytest.approx(expected_mse)
        assert metrics["reconstructed_psnr_db"] == pytest.approx(-10 * np.log10(expected_mse))
        assert metrics["blurred_psnr_db"] == pytest.approx(-10 * np.log10(np.mean((measurement - reference)**2)))
        summary = next(row for row in summaries if row["spatial_prior"] == spatial and row["temporal_prior"] == temporal)
        assert set(summary) == set(metrics)
        assert float(summary["reconstructed_psnr_db"]) == metrics["reconstructed_psnr_db"]
        assert json.loads(summary["shape"]) == list(measurement.shape)
        rows = [row for row in history if row["spatial_prior"] == spatial and row["temporal_prior"] == temporal]
        assert [int(row["iteration"]) for row in rows] == [1, 2, 3, 4]
        np.testing.assert_array_equal([float(row["data_loss"]) for row in rows], np.load(folder / "data_losses.npy"))
        assert all(int(row["prox_iterations"]) > 0 and float(row["prox_gap"]) >= 0 for row in rows)
        assert all(row["prox_converged"] in ("True", "False") for row in rows)
        assert np.load(folder / "data_losses.npy").shape == (4,)
        assert np.load(folder / "reconstructed.npy").shape == measurement.shape
        assert (folder / "reconstructed.mp4").stat().st_size > 0
    assert (comparison / "loss_all_priors.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    single = tmp_path / "single"
    subprocess.run(base + ["--all-priors", "false", "--output-dir", str(single)],
                   cwd=root, env=environment, check=True, capture_output=True, text=True)
    metrics = json.loads((single / "metrics.json").read_text())
    assert metrics["spatial_prior"] == metrics["temporal_prior"] == "tv"
    assert metrics["ground_truth"] == f"{dataset.resolve()}::reference"
    assert metrics["blurred_mse"] == 0
    assert metrics["blurred_psnr_db"] == "Infinity"
    # Ground-truth selection only affects evaluation, never reconstruction.
    assert not list(single.glob("spatial_*"))
    assert not (single / "loss_all_priors.png").exists()
    for name in ("data_losses.npy", "reconstructed.npy"):
        np.testing.assert_array_equal(np.load(single / name),
                                      np.load(comparison / "spatial_tv_temporal_tv" / name))


def test_real_mp4_cli_on_available_device(tmp_path):
    root = Path(__file__).resolve().parents[2]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    clean = np.zeros((13, 17, 3, 3))
    clean[3:10, 4:12] = .7
    source, psf = tmp_path / "input.mp4", tmp_path / "psf.tiff"
    write_video(source, clean, 12)
    iio.imwrite(psf, np.outer([1, 2, 1], [1, 2, 1]).astype(np.float32))
    output = tmp_path / "output"
    subprocess.run([sys.executable, str(root / "torch-implementation/main.py"),
                    str(source), "--psf", str(psf), "--device", device,
                    "--num-iters", "2", "--output-dir", str(output)],
                   check=True, capture_output=True, text=True)
    result = np.load(output / "reconstructed.npy")
    assert result.shape == clean.shape and np.isfinite(result).all()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["device"].startswith(device)
    assert metrics["lambda_spatial"] == metrics["lambda_temporal"] == 1e-5
    assert metrics["reconstructed_psnr_db"] is None
    assert (output / "reconstructed.mp4").stat().st_size > 0


def test_psnr_known_values_and_no_clipping():
    mse, psnr = quality_metrics(np.full((2, 3, 1, 2), .1), np.zeros((2, 3, 1, 2)))
    assert mse == pytest.approx(.01) and psnr == pytest.approx(20)
    assert quality_metrics(np.ones((2, 3)), np.ones((2, 3))) == (0, "Infinity")
    mse, psnr = quality_metrics(np.array([1.2]), np.array([1.0]))
    assert mse == pytest.approx(.04) and psnr == pytest.approx(-10 * np.log10(.04))
    # Pool squared error over the video instead of averaging per-frame dB.
    _, psnr = quality_metrics(np.array([.1, .2]), np.zeros(2))
    assert psnr == pytest.approx(-10 * np.log10(.025))


def test_ground_truth_arrays_and_validation(tmp_path):
    shape = (5, 7, 3, 2)
    reference = np.full(shape, .5)
    path = tmp_path / "gt.npz"
    np.savez(path, reference=reference, fps=12)
    np.testing.assert_array_equal(load_ground_truth(path, shape, 12), reference)
    with pytest.raises(ValueError, match="FPS"):
        load_ground_truth(path, shape, 24)
    np.savez(path, measurement=reference)
    with pytest.raises(ValueError, match="reference"):
        load_ground_truth(path, shape, 12)
    path = tmp_path / "gt.npy"
    for values, message in [(reference[..., :1], "shape"), (reference * 510, "normalized"),
                            (reference * np.nan, "finite")]:
        np.save(path, values)
        with pytest.raises(ValueError, match=message):
            load_ground_truth(path, shape, 12)


def test_ground_truth_mp4_resizing_and_alignment(tmp_path):
    from PIL import Image

    path = tmp_path / "gt.mp4"
    video = np.random.default_rng(7).uniform(size=(12, 16, 3, 3))
    write_video(path, video, 12)
    decoded, _, _ = read_video(path)
    np.testing.assert_array_equal(load_ground_truth(path, video.shape, 12), decoded)
    expected = np.stack([np.asarray(Image.fromarray(np.rint(decoded[..., t] * 255).astype(np.uint8)).resize(
        (8, 6), Image.Resampling.LANCZOS)).astype(float) / 255 for t in range(3)], axis=-1)
    np.testing.assert_array_equal(load_ground_truth(path, (6, 8, 3, 3), 12), expected)
    for shape, fps, message in [((6, 8, 3, 2), 12, "frame counts"),
                               ((6, 8, 3, 4), 12, "frame counts"),
                               ((6, 8, 3, 3), 24, "FPS"),
                               ((24, 32, 3, 3), 12, "smaller")]:
        with pytest.raises(ValueError, match=message):
            load_ground_truth(path, shape, fps)


def test_cli_bad_ground_truth_rejected_before_reconstruction(tmp_path):
    root = Path(__file__).resolve().parents[2]
    dataset, gt, output = tmp_path / "data.npz", tmp_path / "gt.npy", tmp_path / "output"
    np.savez(dataset, measurement=np.zeros((5, 7, 3, 2)), psf=np.ones((1, 1)), fps=12)
    np.save(gt, np.zeros((5, 7, 3, 1)))
    result = subprocess.run([sys.executable, str(root / "torch-implementation/main.py"),
                             str(dataset), "--gt", str(gt), "--output-dir", str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "Ground-truth shape" in result.stderr
    assert not output.exists()
