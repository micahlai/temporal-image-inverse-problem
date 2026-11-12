"""Read sweep histories and exercise actual Matplotlib slider/click callbacks."""

import csv
from itertools import product
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from torch_impl.view_sweep import load_sweep, SweepViewer


def write_data(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def rows():
    return [dict(spatial_prior="tv", temporal_prior="tv", spatial_exponent=se,
                 temporal_exponent=te, iterations=3, iteration=n,
                 reconstructed_psnr_db=score + 2 * si + ti,
                 intensity_constraint="True")
            for (si, se), (ti, te) in product(enumerate([-5.5, -5.25]), enumerate([-5, -4.75]))
            for n, score in enumerate([20., 31., 30.], start=1)]


def test_slider_click_orientation_peak_final_and_fixed_scale(tmp_path, rows):
    path = tmp_path / "history.csv"
    write_data(path, rows)
    data = load_sweep(path)
    assert data.spatial == [-5.5, -5.25] and data.temporal == [-5, -4.75]
    viewer = SweepViewer(data)
    try:
        assert viewer.selected == (1, 1)
        np.testing.assert_allclose(viewer.image.get_array(), [[30, 32], [31, 33]])
        limits = viewer.image.get_clim()
        viewer.slider.set_val(1)
        np.testing.assert_allclose(viewer.image.get_array(), [[20, 22], [21, 23]])
        assert viewer.image.get_clim() == limits == (20., 34.)
        viewer.on_click(SimpleNamespace(inaxes=viewer.heat_axis, button=1, xdata=0., ydata=1.))
        assert viewer.selected == (1, 0)
        np.testing.assert_array_equal(viewer.line.get_ydata(), [21, 32, 31])
        np.testing.assert_array_equal(viewer.peak.get_xdata(), [2])
        np.testing.assert_array_equal(viewer.peak.get_ydata(), [32])
        np.testing.assert_array_equal(viewer.final.get_ydata(), [31])
        np.testing.assert_array_equal(viewer.current.get_ydata(), [21])
        assert "iteration 2" in viewer.detail.get_text()
        viewer.figure.canvas.draw()
    finally:
        plt.close(viewer.figure)


def test_missing_run_is_masked_and_not_selectable(tmp_path, rows):
    path = tmp_path / "history.csv"
    write_data(path, [r for r in rows if not (r["spatial_exponent"] == -5.5 and r["temporal_exponent"] == -5)])
    viewer = SweepViewer(load_sweep(path))
    try:
        assert viewer.image.get_array().mask[0, 0]
        previous = viewer.selected
        viewer.select(0, 0)
        assert viewer.selected == previous
    finally:
        plt.close(viewer.figure)


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows + [rows[0]], "Duplicate"),
    (lambda rows: rows[1:], "Incomplete"),
    (lambda rows: [dict(r, iterations=4) if i == 0 else r for i, r in enumerate(rows)], "mixes"),
    (lambda rows: [dict(r, intensity_constraint="False") if i == 0 else r for i, r in enumerate(rows)], "mixes"),
    (lambda rows: [dict(r, reconstructed_psnr_db="nan") for r in rows], "Invalid"),
])
def test_invalid_histories(tmp_path, rows, change, match):
    path = tmp_path / "history.csv"
    write_data(path, change(rows))
    with pytest.raises(ValueError, match=match):
        load_sweep(path)


@pytest.mark.parametrize("score", [25., float("inf")])
def test_single_iteration_constant_and_infinite_scores(tmp_path, rows, score):
    path = tmp_path / "history.csv"
    write_data(path, [dict(rows[0], iterations=1, reconstructed_psnr_db=score)])
    viewer = SweepViewer(load_sweep(path))
    try:
        assert not viewer.slider.active
        assert np.isfinite(viewer.image.get_array()).all()
        assert np.isfinite(viewer.line.get_ydata()).all()
        if np.isinf(score):
            assert "∞" in viewer.detail.get_text()
        viewer.figure.canvas.draw()
    finally:
        plt.close(viewer.figure)
