"""Tiny synthetic checks only: never launch the real 100-run experiment."""

import argparse
import csv
from itertools import product
import sys

import numpy as np
import pytest

from torch_impl import sweep_weights
from torch_impl.inverse_solver import InverseSolver
from torch_impl.main import run_reconstruction, quality_metrics, write_csv
from torch_impl.plot_sweep_heatmaps import load_grids, plot_heatmaps


def test_grid_has_exactly_100_configurations():
    configs = list(sweep_weights.configurations())
    assert len(configs) == len(set(configs)) == 100
    for pair in sweep_weights.PAIRS:
        assert {(se, te) for s, t, se, te in configs if (s, t) == pair} == set(product(range(-7, -2), repeat=2))


def test_every_iteration_psnr_matches_independent_reconstruction(tmp_path):
    rng = np.random.default_rng(9)
    reference = rng.uniform(size=(3, 5, 3, 2))
    measurement = reference + rng.normal(0, .01, reference.shape)
    args = argparse.Namespace(input=tmp_path / "data.npz", num_iters=3,
        lambda_spatial=.001, lambda_temporal=.0001, wavelet="db4", wavelet_level=4,
        prox_iters=200, prox_tol=1e-5, device="cpu", dtype="float64", ground_truth_source="synthetic")
    _, _, history = run_reconstruction(args, measurement, np.ones((1, 1)), 12,
                                       reference, "tv", "tv", tmp_path, save_artifacts=False)
    for n, row in enumerate(history, start=1):
        solver = InverseSolver([[1]], num_iters=n, lam_spatial=.001, lam_temporal=.0001,
                               spatial_prior="tv", temporal_prior="tv", device="cpu")
        expected = solver(measurement).numpy()
        mse, psnr = quality_metrics(expected, reference)
        assert row["reconstructed_mse"] == pytest.approx(mse, abs=1e-15)
        assert row["reconstructed_psnr_db"] == pytest.approx(psnr, abs=1e-10)
        assert row["data_loss"] == solver.losses[-1]
    assert not list(tmp_path.iterdir())


def test_tiny_sweep_writes_flat_csv_and_heatmaps(tmp_path, monkeypatch):
    dataset = tmp_path / "data.npz"
    reference = np.random.default_rng(1).uniform(size=(3, 5, 3, 2))
    np.savez(dataset, measurement=reference + .01, reference=reference, psf=np.ones((1, 1)), fps=12)
    output = tmp_path / "results"
    # Four tiny 2-iteration runs; the actual sweep defaults are tested separately.
    monkeypatch.setattr(sweep_weights, "configurations", lambda: (
        (s, t, -5, -5) for s, t in sweep_weights.PAIRS))
    monkeypatch.setattr(sys, "argv", ["sweep_weights.py", str(dataset), "--num-iters", "2",
                                     "--device", "cpu", "--output-dir", str(output)])
    sweep_weights.main()
    with (output / "sweep_iterations.csv").open() as stream:
        history = list(csv.DictReader(stream))
    with (output / "sweep_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(history) == 8 and len(summary) == 4
    assert len({row["run_id"] for row in history}) == 4
    assert all(float(row["lambda_spatial"]) == 1e-5 for row in history)
    assert all(float(row["lambda_temporal"]) == 1e-5 for row in history)
    assert {path.name for path in output.iterdir()} == {"sweep_iterations.csv", "sweep_summary.csv"}
    paths = plot_heatmaps(output / "sweep_iterations.csv", output)
    assert len(paths) == 1
    assert paths[0].name == "psnr_all_priors_iter_2.png"
    assert list(output.glob("*.png")) == paths
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)
    xs, ys, from_history, selected = load_grids(output / "sweep_iterations.csv")
    _, _, from_summary, _ = load_grids(output / "sweep_summary.csv")
    assert xs == ys == [-5] and selected == 2
    for pair in sweep_weights.PAIRS:
        np.testing.assert_allclose(from_history[pair], from_summary[pair], atol=1e-10)
    before = (output / "sweep_iterations.csv").read_bytes()
    with pytest.raises(SystemExit):
        sweep_weights.main()
    assert (output / "sweep_iterations.csv").read_bytes() == before


def test_heatmap_orientation_iteration_and_invalid_data(tmp_path):
    rows = [dict(spatial_prior=s, temporal_prior=t, spatial_exponent=se,
                 temporal_exponent=te, iterations=250, iteration=n,
                 reconstructed_psnr_db=100 + se + 10 * te + n / 100)
            for s, t in sweep_weights.PAIRS for se, te in product([-7, -6], [-5, -4])
            for n in (1, 250)]
    path = tmp_path / "sweep.csv"
    write_csv(path, rows)
    xs, ys, grids, n = load_grids(path)
    assert xs == [-7, -6] and ys == [-5, -4] and n == 250
    np.testing.assert_allclose(grids[("tv", "wavelet")], [[45.5, 46.5], [55.5, 56.5]])
    assert load_grids(path, 1)[2][("tv", "tv")][0, 0] == pytest.approx(43.01)
    with pytest.raises(ValueError, match="missing"):
        load_grids(path, 2)
    write_csv(path, rows + [rows[-1]])
    with pytest.raises(ValueError, match="Duplicate"):
        load_grids(path)
    write_csv(path, rows[:-1])
    with pytest.raises(ValueError, match="missing"):
        load_grids(path)


@pytest.mark.parametrize("values,limits", [
    ([10., 20., 30., 40.], (10., 40.)),
    ([25., 25., 25., 25.], (24.5, 25.5)),
    ([10., 20., 30., float("inf")], (10., 30.)),
    ([float("inf")] * 4, (0., 1.)),
])
def test_four_heatmaps_share_color_limits_and_mapping(tmp_path, monkeypatch, values, limits):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    path = tmp_path / "colors.csv"
    write_csv(path, [dict(spatial_prior=s, temporal_prior=t, spatial_exponent=-5,
                         temporal_exponent=-5, iterations=250, reconstructed_psnr_db=value)
                     for (s, t), value in zip(sweep_weights.PAIRS, values)])
    images = []
    original_imshow = Axes.imshow

    def capture_image(axis, *args, **kwargs):
        image = original_imshow(axis, *args, **kwargs)
        images.append(image)
        return image

    monkeypatch.setattr(Axes, "imshow", capture_image)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    plot_heatmaps(path, tmp_path)
    assert len(images) == 4
    figure = images[0].figure
    assert all(image.figure is figure for image in images)
    assert len(figure.axes) == 5  # Four panels and exactly one colorbar.
    colorbars = [image.colorbar for image in images if image.colorbar is not None]
    assert len(colorbars) == 1
    assert colorbars[0].mappable.norm is images[0].norm
    assert {image.axes.get_subplotspec().num1 for image in images} == {0, 1, 2, 3}
    probes = np.array([limits[0], sum(limits) / 2, limits[1], float("inf")])
    for image in images:
        assert image.get_clim() == limits
        assert image.norm is images[0].norm
        np.testing.assert_array_equal(image.to_rgba(probes), images[0].to_rgba(probes))
