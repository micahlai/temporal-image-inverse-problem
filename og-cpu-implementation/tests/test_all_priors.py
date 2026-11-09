"""CLI comparison behavior and Matplotlib axis/curve verification."""

import itertools
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

from lab8_temporal.main import plot_prior_losses


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
    environment = dict(os.environ, MPLBACKEND="Agg")
    base = [sys.executable, "-m", "lab8_temporal.main", str(dataset),
            "--spatial-prior", "tv", "--temporal-prior", "tv",
            "--lambda-spatial", "-2", "--lambda-temporal", "-3",
            "--num-iters", "4", "--show-plot", "false"]
    comparison = tmp_path / "all"
    subprocess.run(base + ["--all-priors", "true", "--output-dir", str(comparison)],
                   cwd=root, env=environment, check=True, capture_output=True, text=True)
    folders = list(comparison.glob("spatial_*_temporal_*"))
    assert len(folders) == 4
    for spatial, temporal in itertools.product(("wavelet", "tv"), repeat=2):
        folder = comparison / f"spatial_{spatial}_temporal_{temporal}"
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["spatial_prior"] == spatial
        assert metrics["temporal_prior"] == temporal
        assert metrics["lambda_spatial"] == 1e-2
        assert metrics["lambda_temporal"] == 1e-3
        assert np.load(folder / "data_losses.npy").shape == (4,)
        assert np.load(folder / "reconstructed.npy").shape == measurement.shape
        assert (folder / "reconstructed.mp4").stat().st_size > 0
    assert (comparison / "loss_all_priors.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    single = tmp_path / "single"
    subprocess.run(base + ["--all-priors", "false", "--output-dir", str(single)],
                   cwd=root, env=environment, check=True, capture_output=True, text=True)
    metrics = json.loads((single / "metrics.json").read_text())
    assert metrics["spatial_prior"] == metrics["temporal_prior"] == "tv"
    assert not list(single.glob("spatial_*"))
    assert not (single / "loss_all_priors.png").exists()
    for name in ("data_losses.npy", "reconstructed.npy"):
        np.testing.assert_array_equal(np.load(single / name),
                                      np.load(comparison / "spatial_tv_temporal_tv" / name))
