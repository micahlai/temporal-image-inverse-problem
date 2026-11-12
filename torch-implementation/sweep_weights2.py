"""Sweep 25 TV/TV weight pairs with a [0,1] intensity constraint; CSV output only."""

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

try:
    from .main import load_ground_truth, validate_reference, run_reconstruction, write_csv
except ImportError:
    from main import load_ground_truth, validate_reference, run_reconstruction, write_csv


HERE = Path(__file__).resolve().parent
PAIRS = (("tv", "tv"),)
EXPONENTS = (-5.5, -5.25, -5, -4.75, -4.5)


def configurations():
    """All 25 spatial/temporal exponent pairs, using TV in both dimensions."""
    for spatial, temporal in PAIRS:
        for spatial_exp, temporal_exp in product(EXPONENTS, repeat=2):
            yield spatial, temporal, spatial_exp, temporal_exp


def checkpoint(path, records):
    temporary = path.with_suffix(".csv.tmp")
    write_csv(temporary, records)
    temporary.replace(path)


def configuration_id(config):
    spatial, temporal, se, te = config
    return f"spatial_{spatial}_{se}_temporal_{temporal}_{te}"


def load_checkpoint(history_path, summary_path, configs, start_run, args):
    """Preserve saved rows; require complete runs before the restart point."""
    if start_run == 1:
        if history_path.exists() or summary_path.exists():
            raise ValueError("Sweep CSVs already exist; use a new --output-dir for --start-run 1")
        return [], []
    if not history_path.exists() or not summary_path.exists():
        raise ValueError("Restart requires both existing sweep CSVs; use --start-run 1 for a new sweep")
    with history_path.open(newline="", encoding="utf-8") as stream:
        histories = list(csv.DictReader(stream))
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    ids = [configuration_id(config) for config in configs]
    if any(row.get("run_id") not in ids for row in histories + summaries):
        raise ValueError("Checkpoint contains runs outside this sweep's weight grid")
    for run_id in ids[:start_run - 1]:
        reports = [row for row in summaries if row["run_id"] == run_id]
        rows = [row for row in histories if row["run_id"] == run_id]
        if len(reports) != 1 or [int(row["iteration"]) for row in rows] != list(range(1, args.num_iters + 1)):
            raise ValueError(f"Checkpoint is incomplete for {run_id}; restart at that run or earlier")
        expected = dict(input=str(args.input.resolve()), ground_truth=args.ground_truth_source,
                        iterations=args.num_iters, prox_iters=args.prox_iters,
                        wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                        intensity_constraint=True)
        report = reports[0]
        if any(report.get(key) != str(value) for key, value in expected.items()) or float(report["prox_tol"]) != args.prox_tol:
            raise ValueError(f"Checkpoint settings differ for {run_id}; use the original input and solver settings")
    return histories, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?",
                        default=HERE.parent / "og-cpu-implementation/data/smol_noise/dataset.npz",
                        help="NPZ dataset or directory containing dataset.npz")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/smol_noise_sweep2")
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--num-iters", type=int, default=2000,
                        help="Outer iterations per run (default: 2000)")
    parser.add_argument("--prox-iters", type=int, default=200)
    parser.add_argument("--prox-tol", type=float, default=1e-5)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--start-run", type=int, default=1,
                        help="Start or restart at this 1-based run, inclusive (default: 1); e.g. 21 to resume at 21/25")
    parser.set_defaults(intensity_constraint=True)
    args = parser.parse_args()
    if args.input.is_dir():
        args.input = args.input / "dataset.npz"
    history_path = args.output_dir / "sweep_iterations.csv"
    summary_path = args.output_dir / "sweep_summary.csv"
    configs = list(configurations())
    if not 1 <= args.start_run <= len(configs):
        parser.error(f"--start-run must be between 1 and {len(configs)}")
    if args.num_iters < 1:
        parser.error("--num-iters must be positive")
    try:
        with np.load(args.input, allow_pickle=False) as dataset:
            measurement, psf, fps = dataset["measurement"], dataset["psf"], float(dataset["fps"])
            reference = dataset["reference"] if "reference" in dataset else None
        if args.gt is not None:
            reference = load_ground_truth(args.gt, measurement.shape, fps)
            args.ground_truth_source = str(args.gt.resolve())
        elif reference is not None:
            validate_reference(reference, measurement.shape)
            args.ground_truth_source = f"{args.input.resolve()}::reference"
        else:
            parser.error("PSNR sweep requires dataset reference or --gt")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    try:
        histories, summaries = load_checkpoint(history_path, summary_path, configs, args.start_run, args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(configs)} constrained TV/TV runs x {args.num_iters} iterations; CSV directory: {args.output_dir}", flush=True)
    print(f"Starting at run {args.start_run}/{len(configs)}; earlier runs are preserved.", flush=True)
    for number, config in enumerate(configs[args.start_run - 1:], start=args.start_run):
        spatial, temporal, se, te = config
        args.lambda_spatial, args.lambda_temporal = 10.0 ** se, 10.0 ** te
        run_id = configuration_id(config)
        print(f"Run {number}/{len(configs)}: {run_id}", flush=True)
        _, report, history = run_reconstruction(
            args, measurement, psf, fps, reference, spatial, temporal,
            args.output_dir, save_artifacts=False)
        metadata = dict(run_id=run_id, spatial_exponent=se, temporal_exponent=te)
        # Remove a previous/partial copy only after its replacement finishes.
        # This prevents duplicate rows and leaves disk data intact on Ctrl-C.
        summaries = [row for row in summaries if row["run_id"] != run_id]
        histories = [row for row in histories if row["run_id"] != run_id]
        summaries.append({**metadata, **report})
        # Each iteration row is self-contained: settings, baseline, diagnostics,
        # loss, and PSNR. Per-iteration scores override the final report scores.
        histories.extend({**metadata, **report, **row} for row in history)
        checkpoint(history_path, histories)
        checkpoint(summary_path, summaries)
        print(f"Saved {number}/{len(configs)} completed runs", flush=True)
    print(f"Done: {len(histories)} iteration rows and {len(summaries)} summary rows.")


if __name__ == "__main__":
    main()
