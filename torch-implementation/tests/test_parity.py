"""Compare the actual original implementation with the port, including GPU."""

import itertools

import numpy as np
import pytest
import torch
from scipy.signal import convolve2d

from cpu_reference.inverse_solver import InverseSolver as CPUSolver
from cpu_reference import priors as cpu
from torch_impl.backend import resolve_device, resolve_dtype
from torch_impl.inverse_solver import InverseSolver, adjoint_test
from torch_impl.priors import _wavelet, joint_prox, prox_tv, prox_wavelet


PAIRS = list(itertools.product(("wavelet", "tv"), repeat=2))
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable")),
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="NVIDIA GPU unavailable"))]


def host(x):
    return x.detach().cpu().numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(7, 9, 2, 5), (8, 6, 1, 1), (1, 1, 2, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 5), (4, 2), (11, 13)])
def test_convolution_and_exact_adjoint(device, shape, kernel_shape):
    rng = np.random.default_rng(7)
    kernel, x = rng.uniform(size=kernel_shape), rng.normal(size=shape)
    solver = InverseSolver(kernel, device=device)
    actual = solver.A(x)
    atol = 2e-6 if device != "cpu" else 1e-12
    assert actual.device.type == device
    np.testing.assert_allclose(host(actual), CPUSolver(kernel).A(x), atol=atol)
    full = convolve2d(x[:, :, 0, 0], kernel / kernel.sum(), mode="full")
    kh, kw = kernel_shape
    expected = full[kh // 2:kh // 2 + shape[0], kw // 2:kw // 2 + shape[1]]
    np.testing.assert_allclose(host(actual)[:, :, 0, 0], expected, atol=atol)
    np.testing.assert_allclose(*adjoint_test(solver, shape), atol=atol * 10, rtol=atol * 10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym3", "coif1"])
@pytest.mark.parametrize("axes", [(0, 1), (3,)])
@pytest.mark.parametrize("shape", [(7, 9, 2, 5), (8, 6, 1, 8)])
def test_wavelet_matches_original_and_is_orthonormal(device, wavelet, axes, shape):
    rng = np.random.default_rng(8)
    x = rng.normal(size=shape)
    dtype = torch.float64 if device == "cpu" else torch.float32
    tensor = torch.tensor(x, dtype=dtype, device=device)
    transformed = _wavelet(tensor, axes, wavelet, 4)
    tol = 2e-6 if device != "cpu" else 2e-10  # sym3's published filter precision
    np.testing.assert_allclose(host(transformed), cpu._wavelet(x, axes, wavelet, 4), atol=tol, rtol=tol)
    np.testing.assert_allclose(host(_wavelet(transformed, axes, wavelet, 4, True)), x, atol=tol, rtol=tol)
    np.testing.assert_allclose(host(transformed.square().sum()), np.square(x).sum(), rtol=tol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spatial,temporal", PAIRS)
def test_joint_prox_parity_and_stopping_rule(device, spatial, temporal):
    x = np.random.default_rng(5).normal(size=(5, 7, 2, 3))
    dtype = torch.float64 if device == "cpu" else torch.float32
    tensor = torch.tensor(x, device=device, dtype=dtype)
    expected, expected_info = cpu.joint_prox(x, .03, .06, .7, spatial, temporal, return_info=True)
    actual, info = joint_prox(tensor, .03, .06, .7, spatial, temporal, return_info=True)
    tol = 2e-6 if device != "cpu" else 1e-12
    np.testing.assert_allclose(host(actual), expected, atol=tol, rtol=tol)
    assert info["iterations"] == expected_info["iterations"]
    assert info["converged"] == expected_info["converged"]
    assert actual.device.type == device and not actual.requires_grad
    np.testing.assert_array_equal(host(joint_prox(tensor, 0, 0, 1, spatial, temporal)), host(tensor))
    for weights in [(0, .06), (.03, 0)]:
        assert not torch.allclose(actual, joint_prox(tensor, *weights, .7, spatial, temporal))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spatial,temporal", PAIRS)
@pytest.mark.parametrize("algorithm", ["fista", "ista", "gradient_descent"])
def test_full_solver_matches_original(device, spatial, temporal, algorithm):
    rng = np.random.default_rng(19)
    psf = rng.uniform(size=(4, 3))
    y = rng.normal(size=(7, 9, 2, 5))
    options = dict(num_iters=5, lam_spatial=.004, lam_temporal=.007,
                   spatial_prior=spatial, temporal_prior=temporal, algorithm=algorithm)
    original, port = CPUSolver(psf, **options), InverseSolver(psf, device=device, **options)
    callbacks = []
    expected = original(y)
    actual = port(y, callback=lambda n, loss: callbacks.append((n, loss)))
    tol = 5e-6 if device != "cpu" else 1e-11
    np.testing.assert_allclose(host(actual), expected, atol=tol, rtol=tol)
    np.testing.assert_allclose(port.losses, original.losses, atol=tol, rtol=tol)
    assert callbacks == list(enumerate(port.losses, start=1))
    assert actual.device.type == device
    assert actual.min() < 0  # No new positivity/clipping constraint.


@pytest.mark.parametrize("device", DEVICES)
def test_temporal_boundaries_and_color_isolation(device):
    dtype = torch.float64 if device == "cpu" else torch.float32
    x = torch.zeros((3, 5, 2, 2), device=device, dtype=dtype)
    x[:, :, 0, 1], x[:, :, 1, :] = 1, 8
    actual = prox_tv(x, .2, 1, temporal=True, max_iters=1000, tol=1e-7)
    np.testing.assert_allclose(host(actual[:, :, 0, 0]), .2, atol=2e-6)
    np.testing.assert_allclose(host(actual[:, :, 0, 1]), .8, atol=2e-6)
    np.testing.assert_allclose(host(actual[:, :, 1, :]), 8, atol=2e-6)
    torch.testing.assert_close(prox_tv(x, .2, 1), x)
    singleton = x[..., 1:]
    torch.testing.assert_close(prox_wavelet(singleton, .2, 1, temporal=True), singleton)
    impulse = torch.zeros_like(x)
    impulse[1, 2, 0, 0] = 1
    blurred = InverseSolver(np.ones((3, 3)), device=device).A(impulse)
    torch.testing.assert_close(blurred[:, :, 1, :], torch.zeros_like(blurred[:, :, 1, :]))
    torch.testing.assert_close(blurred[:, :, :, 1], torch.zeros_like(blurred[:, :, :, 1]))


@pytest.mark.parametrize("spatial,temporal", PAIRS)
def test_reconstruction_improves_synthetic_mse(spatial, temporal):
    clean = np.zeros((15, 17, 3, 5))
    clean[4:11, 5:12] = np.array([.4, .7, 1])[None, None, :, None]
    solver = InverseSolver(np.outer([1, 2, 1], [1, 2, 1]), device="cpu", num_iters=30,
                           spatial_prior=spatial, temporal_prior=temporal,
                           lam_spatial=1e-4, lam_temporal=1e-4, prox_iters=500)
    y = solver.A(clean)
    result = host(solver(y))
    assert np.mean((result - clean)**2) < np.mean((host(y) - clean)**2)


def test_validation_precision_and_iteration_cap():
    with pytest.raises(ValueError, match="float64"):
        resolve_dtype("float64", torch.device("mps"))
    with pytest.raises(ValueError, match="dtype"):
        resolve_dtype("float16", torch.device("cpu"))
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    with pytest.raises(ValueError, match="orthogonal"):
        joint_prox(np.ones((3, 3, 1, 2)), .1, .1, 1, wavelet="bior2.2")
    for psf in [np.zeros((2, 2)), np.array([[-1, 2]]), np.array([[np.nan]])]:
        with pytest.raises(ValueError, match="psf"):
            InverseSolver(psf, device="cpu")
    with pytest.raises(ValueError, match="finite"):
        InverseSolver([[1]], device="cpu")(np.full((2, 2, 1, 2), np.nan))
    solver = InverseSolver([[1]], device="cpu", num_iters=1, prox_iters=1,
                           spatial_prior="tv", temporal_prior="tv", prox_tol=1e-15)
    with pytest.warns(RuntimeWarning, match="prox_iters"):
        solver(np.random.default_rng(0).normal(size=(5, 7, 2, 3)))
    assert not solver.prox_diagnostics[0]["converged"]
