"""Plot one 2x2 PSNR heatmap figure from sweep CSVs, with a shared colorbar."""

import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


PAIRS = tuple(product(("wavelet", "tv"), repeat=2))


def load_grids(path, iteration=None):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"spatial_prior", "temporal_prior", "spatial_exponent", "temporal_exponent",
                    "iterations", "reconstructed_psnr_db"}
        if not required <= fields:
            raise ValueError(f"Missing CSV columns: {sorted(required - fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data")
    try:
        totals = {int(row["iterations"]) for row in rows}
        if len(totals) != 1:
            raise ValueError("CSV mixes different run lengths")
        total = totals.pop()
        selected_iteration = total if iteration is None else iteration
        if not 1 <= selected_iteration <= total:
            raise ValueError(f"Iteration must be between 1 and {total}")
        if "iteration" in fields:
            selected = [row for row in rows if int(row["iteration"]) == selected_iteration]
        else:
            if selected_iteration != total:
                raise ValueError("Use sweep_iterations.csv to select an intermediate iteration")
            selected = rows
        xs = sorted({int(row["spatial_exponent"]) for row in rows})
        ys = sorted({int(row["temporal_exponent"]) for row in rows})
        grids = {pair: np.full((len(ys), len(xs)), np.nan) for pair in PAIRS}
        seen = set()
        for row in selected:
            pair = row["spatial_prior"], row["temporal_prior"]
            se, te = int(row["spatial_exponent"]), int(row["temporal_exponent"])
            key = (*pair, se, te)
            if key in seen:
                raise ValueError(f"Duplicate heatmap cell: {key}")
            if pair not in grids:
                raise ValueError(f"Unknown prior combination: {pair}")
            value = float(row["reconstructed_psnr_db"])
            if np.isnan(value) or np.isneginf(value):
                raise ValueError("PSNR must be finite or positive Infinity")
            grids[pair][ys.index(te), xs.index(se)] = value
            seen.add(key)
        if len(seen) != 4 * len(xs) * len(ys):
            raise ValueError("Incomplete sweep at the requested iteration; some heatmap cells are missing")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid sweep CSV: {error}") from error
    return xs, ys, grids, selected_iteration


def plot_heatmaps(path, output_dir, iteration=None):
    xs, ys, grids, selected_iteration = load_grids(path, iteration)
    all_values = np.concatenate([grid.ravel() for grid in grids.values()])
    finite = all_values[np.isfinite(all_values)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
    if low == high:
        low, high = low - .5, high + .5
    norm = Normalize(low, high)
    # Exact-match Infinity is distinct from finite maxima.
    cmap = plt.get_cmap("viridis").with_extremes(over="#ff66cc")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    figure.suptitle(f"PSNR at iteration {selected_iteration}", fontsize=16)
    for axis, ((spatial, temporal), grid) in zip(axes.flat, grids.items()):
        colors = np.where(np.isposinf(grid), high + (high - low), grid)
        plotted = axis.imshow(colors, origin="lower", cmap=cmap, norm=norm, aspect="equal")
        axis.set_xticks(range(len(xs)), [f"$10^{{{x}}}$" for x in xs])
        axis.set_yticks(range(len(ys)), [f"$10^{{{y}}}$" for y in ys])
        axis.set_xlabel("Spatial weight")
        axis.set_ylabel("Temporal weight")
        axis.set_title(f"Spatial {spatial} / temporal {temporal}")
        for y, x in np.ndindex(grid.shape):
            value = grid[y, x]
            label = "∞" if np.isposinf(value) else f"{value:.2f}"
            axis.text(x, y, label, ha="center", va="center",
                      color="white" if np.isfinite(value) and norm(value) < .5 else "black")
    figure.colorbar(plotted, ax=axes.ravel().tolist(), label="PSNR (dB)",
                    shrink=.85, extend="max" if np.isposinf(all_values).any() else "neither")
    if np.isposinf(all_values).any():
        figure.supxlabel("Pink / ∞ denotes an exact match")
    destination = output_dir / f"psnr_all_priors_iter_{selected_iteration}.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return [destination]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Default: same directory as CSV")
    parser.add_argument("--iteration", type=int, help="Default: final iteration (250 for the standard sweep)")
    args = parser.parse_args()
    try:
        for path in plot_heatmaps(args.csv, args.output_dir or args.csv.parent, args.iteration):
            print(f"Saved: {path}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
