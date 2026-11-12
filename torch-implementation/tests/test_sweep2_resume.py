"""Restart checks use mocked reconstructions and temporary CSVs only."""

import csv
import sys

import numpy as np
import pytest

from torch_impl import sweep_weights2 as sweep


@pytest.mark.parametrize("interrupt", [False, True])
def test_explicit_restart_at_21_preserves_previous_rows(tmp_path, monkeypatch, interrupt):
    dataset = tmp_path / "dataset.npz"
    reference = np.zeros((1, 1, 1, 2))
    np.savez(dataset, measurement=reference, reference=reference, psf=[[1]], fps=12)
    output = tmp_path / "results"
    output.mkdir()
    histories, summaries = [], []
    configs = list(sweep.configurations())
    for config in configs[:21]:
        row = dict(run_id=sweep.configuration_id(config), input=str(dataset.resolve()),
                   spatial_exponent=config[2], temporal_exponent=config[3],
                   ground_truth=f"{dataset.resolve()}::reference", iterations=2,
                   prox_iters=200, prox_tol=1e-5, wavelet="db4", wavelet_level=4,
                   intensity_constraint=True)
        summaries.append(row)
        histories.extend({**row, "iteration": n} for n in (1, 2))
    history_path, summary_path = output / "sweep_iterations.csv", output / "sweep_summary.csv"
    sweep.checkpoint(history_path, histories)
    sweep.checkpoint(summary_path, summaries)
    before = [p.read_bytes() for p in (history_path, summary_path)]
    with history_path.open() as stream:
        previous_rows = list(csv.DictReader(stream))[:40]
    calls = []

    def fake_run(args, *rest, **kwargs):
        calls.append((args.lambda_spatial, args.lambda_temporal))
        if interrupt:
            raise KeyboardInterrupt
        return [], {"iterations": 2}, [{"iteration": 1}, {"iteration": 2}]

    monkeypatch.setattr(sweep, "run_reconstruction", fake_run)
    monkeypatch.setattr(sys, "argv", ["sweep_weights2.py", str(dataset), "--num-iters", "2",
                                     "--output-dir", str(output), "--start-run", "21"])
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            sweep.main()
        assert [p.read_bytes() for p in (history_path, summary_path)] == before
        assert calls == [(10. ** -4.5, 10. ** -5.5)]
    else:
        sweep.main()
        assert calls == [(10. ** se, 10. ** te) for _, _, se, te in configs[20:]]
        with history_path.open() as stream:
            rows = list(csv.DictReader(stream))
        with summary_path.open() as stream:
            reports = list(csv.DictReader(stream))
        assert rows[:40] == previous_rows
        assert len(rows) == 50 and len(reports) == 25
        assert len({row["run_id"] for row in reports}) == 25
