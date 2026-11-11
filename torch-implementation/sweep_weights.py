"""Sweep 5 spatial weights x 5 temporal weights x 4 priors; CSV output only."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

try:
    from .main import load_ground_truth, validate_reference, run_reconstruction, write_csv
except ImportError:
    from main import load_ground_truth, validate_reference, run_reconstruction, write_csv


HERE = Path(__file__).resolve().parent
PAIRS = tuple(product(("wavelet", "tv"), repeat=2))


def configurations():
    """Exactly 100 distinct configurations, with absolute and exponent weights."""
    for spatial, temporal in PAIRS:
        for spatial_exp, temporal_exp in product(range(-7, -2), repeat=2):
            yield spatial, temporal, spatial_exp, temporal_exp


def checkpoint(path, records):
    temporary = path.with_suffix(".csv.tmp")
    write_csv(temporary, records)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?",
                        default=HERE.parent / "og-cpu-implementation/data/smol_noise/dataset.npz",
                        help="NPZ dataset or directory containing dataset.npz")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/smol_noise_sweep")
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--num-iters", type=int, default=250,
                        help="Outer iterations per run (default: 250)")
    parser.add_argument("--prox-iters", type=int, default=200)
    parser.add_argument("--prox-tol", type=float, default=1e-5)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    args = parser.parse_args()
    if args.input.is_dir():
        args.input = args.input / "dataset.npz"
    history_path = args.output_dir / "sweep_iterations.csv"
    summary_path = args.output_dir / "sweep_summary.csv"
    if any(path.exists() for path in (history_path, summary_path)):
        parser.error("Sweep CSVs already exist; select a new --output-dir to preserve completed results")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories, summaries = [], []
    print(f"100 runs x {args.num_iters} iterations; CSV directory: {args.output_dir}", flush=True)
    for number, (spatial, temporal, se, te) in enumerate(configurations(), start=1):
        args.lambda_spatial, args.lambda_temporal = 10.0 ** se, 10.0 ** te
        run_id = f"spatial_{spatial}_{se}_temporal_{temporal}_{te}"
        print(f"Run {number}/100: {run_id}", flush=True)
        _, report, history = run_reconstruction(
            args, measurement, psf, fps, reference, spatial, temporal,
            args.output_dir, save_artifacts=False)
        metadata = dict(run_id=run_id, spatial_exponent=se, temporal_exponent=te)
        summaries.append({**metadata, **report})
        # Each iteration row is self-contained: settings, baseline, diagnostics,
        # loss, and PSNR. Per-iteration scores override the final report scores.
        histories.extend({**metadata, **report, **row} for row in history)
        checkpoint(history_path, histories)
        checkpoint(summary_path, summaries)
        print(f"Saved {number}/100 completed runs", flush=True)
    print(f"Done: {len(histories)} iteration rows and {len(summaries)} summary rows.")


if __name__ == "__main__":
    main()
