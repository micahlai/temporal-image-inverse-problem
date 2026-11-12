"""Bounded joint prox against an independent quadratic program; tiny sweeps."""

import csv
import json
from itertools import product
import sys

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize
import torch

from cpu_reference.priors import _wavelet
from torch_impl import main as main_module, sweep_weights2
from torch_impl.inverse_solver import InverseSolver
from torch_impl.priors import joint_prox


@pytest.mark.parametrize("spatial,temporal", list(product(("tv", "wavelet"), repeat=2)))
def test_constrained_joint_prox_matches_independent_qp(spatial, temporal):
    shape = (2, 2, 1, 2)
    x = np.array([-.8, 1.8, .3, .1, 1.2, -.4, .9, .7]).reshape(shape)
    identity = np.eye(x.size)
    blocks, weights = [], []
    for prior, axes, weight in ((spatial, (0, 1), .3), (temporal, (3,), .2)):
        if prior == "tv":
            transforms = [lambda v, axis=axis: np.diff(v, axis=axis) for axis in axes]
        else:
            transforms = [lambda v, axes=axes: _wavelet(v, axes, "haar", 1)]
        for transform in transforms:
            block = np.column_stack([transform(v.reshape(shape)).ravel() for v in identity])
            blocks.append(block)
            weights.extend([weight] * len(block))
    k = np.vstack(blocks)
    weights = np.array(weights)
    n, m = x.size, len(k)
    # Epigraph formulation: -u <= Kz <= u, 0 <= z <= 1, u >= 0.
    constraint = LinearConstraint(np.block([[k, -np.eye(m)], [-k, -np.eye(m)]]),
                                  -np.inf, 0)
    bounds = Bounds(np.zeros(n + m), np.r_[np.ones(n), np.full(m, np.inf)])
    start = x.ravel().clip(0, 1)
    optimum = minimize(lambda v: .5 * np.sum((v[:n] - x.ravel())**2) + weights @ v[n:],
                       np.r_[start, np.abs(k @ start)],
                       jac=lambda v: np.r_[v[:n] - x.ravel(), weights],
                       bounds=bounds, constraints=constraint, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 1000})
    assert optimum.success, optimum.message
    result, info = joint_prox(x, .3, .2, 1, spatial, temporal, wavelet="haar", level=1,
                              intensity_constraint=True, max_iters=5000, tol=1e-10,
                              return_info=True)
    assert info["converged"]
    np.testing.assert_allclose(result.numpy().ravel(), optimum.x[:n], atol=2e-5)
    assert result.min() >= 0 and result.max() <= 1


@pytest.mark.parametrize("algorithm", ["fista", "ista", "gradient_descent"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable"))])
def test_bounded_iterates_and_unchanged_measurements(algorithm, device):
    measurement = np.array([-1., .2, .8, 2.]).reshape(2, 1, 1, 2)
    original = measurement.copy()
    solver = InverseSolver([[1]], algorithm=algorithm, num_iters=4,
                           lam_spatial=0, lam_temporal=0, device=device,
                           intensity_constraint=True)
    iterates = []
    result = solver(measurement, iterate_callback=lambda n, loss, x: iterates.append(x.cpu().numpy().copy()))
    for iterate in iterates:
        np.testing.assert_allclose(iterate, measurement.clip(0, 1), atol=1e-6)
    np.testing.assert_array_equal(measurement, original)
    assert solver.losses[-1] == pytest.approx(1.)  # Uses the original noisy measurement.
    assert result.device.type == device
    baseline = InverseSolver([[1]], num_iters=1, lam_spatial=0, lam_temporal=0,
                              device="cpu")(measurement)
    np.testing.assert_allclose(baseline.numpy(), original)


def test_sweep2_grid():
    configs = list(sweep_weights2.configurations())
    assert len(configs) == len(set(configs)) == 25
    assert {(s, t) for s, t, _, _ in configs} == {("tv", "tv")}
    assert {(se, te) for _, _, se, te in configs} == set(product(
        (-5.5, -5.25, -5, -4.75, -4.5), repeat=2))


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_main_intensity_constraint_flag(tmp_path, monkeypatch, enabled):
    dataset = tmp_path / "dataset.npz"
    measurement = np.array([-.2, 1.2]).reshape(1, 1, 1, 2)
    np.savez(dataset, measurement=measurement, psf=[[1]], fps=12)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["main.py", str(dataset), "--num-iters", "1",
                                     "--device", "cpu", "--spatial-prior", "tv",
                                     "--temporal-prior", "tv", "--output-dir", str(output),
                                     "--intensity-constraint", enabled])
    monkeypatch.setattr(main_module, "write_video", lambda *args: None)
    main_module.main()
    result = np.load(output / "reconstructed.npy")
    report = json.loads((output / "metrics.json").read_text())
    assert report["intensity_constraint"] == (enabled == "true")
    if enabled == "true":
        np.testing.assert_allclose(result, measurement.clip(0, 1), atol=1e-8)
    else:
        assert result.min() < 0 and result.max() > 1


def test_sweep2_default_settings_without_running_reconstruction(tmp_path, monkeypatch):
    reference = np.full((2, 2, 1, 2), .5)
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, measurement=reference, reference=reference, psf=[[1]], fps=12)
    monkeypatch.setattr(sweep_weights2, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["sweep_weights2.py", str(dataset), "--start-run", "1"])
    calls = []

    def record(args, measurement, psf, fps, truth, spatial, temporal, output, **kwargs):
        assert args.intensity_constraint is True and args.num_iters == 2000
        assert spatial == temporal == "tv"
        assert output == tmp_path / "results/smol_noise_sweep2"
        assert kwargs == {"save_artifacts": False}
        np.testing.assert_array_equal(measurement, reference)
        calls.append((args.lambda_spatial, args.lambda_temporal))
        return [], {}, []

    monkeypatch.setattr(sweep_weights2, "run_reconstruction", record)
    sweep_weights2.main()
    assert set(calls) == set(product([10.**e for e in sweep_weights2.EXPONENTS], repeat=2))


def test_tiny_sweep2_csv_and_overwrite_protection(tmp_path, monkeypatch):
    reference = np.random.default_rng(1).uniform(size=(3, 3, 1, 2))
    measurement = reference.copy()
    measurement[0, 0, 0, :] = [-.2, 1.2]
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, measurement=measurement, reference=reference, psf=[[1]], fps=12)
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["sweep_weights2.py", str(dataset), "--num-iters", "2",
                                     "--device", "cpu", "--output-dir", str(output), "--start-run", "1"])
    sweep_weights2.main()
    history_path = output / "sweep_iterations.csv"
    with history_path.open() as stream:
        history = list(csv.DictReader(stream))
    with (output / "sweep_summary.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(history) == 50 and len(summaries) == 25
    for row in history + summaries:
        assert row["intensity_constraint"] == "True"
        assert row["spatial_prior"] == row["temporal_prior"] == "tv"
        assert float(row["lambda_spatial"]) == pytest.approx(10.**float(row["spatial_exponent"]))
        assert float(row["lambda_temporal"]) == pytest.approx(10.**float(row["temporal_exponent"]))
    assert {p.name for p in output.iterdir()} == {"sweep_summary.csv", "sweep_iterations.csv"}
    before = history_path.read_bytes()
    with pytest.raises(SystemExit):
        sweep_weights2.main()
    assert history_path.read_bytes() == before
