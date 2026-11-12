"""Interactive lambda heatmap and PSNR history from a sweep_iterations.csv."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


@dataclass
class Sweep:
    spatial: list
    temporal: list
    iterations: np.ndarray
    psnr: np.ndarray  # temporal, spatial, iteration; missing runs are NaN.
    pair: tuple
    constraint: str


def load_sweep(path, spatial_prior="tv", temporal_prior="tv"):
    """Read one prior pair; accept incomplete grids, but not incomplete runs."""
    required = {"spatial_prior", "temporal_prior", "spatial_exponent",
                "temporal_exponent", "iterations", "iteration", "reconstructed_psnr_db"}
    runs, totals, constraints = {}, set(), set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Use sweep_iterations.csv with per-iteration PSNR, not sweep_summary.csv")
        for line, row in enumerate(reader, start=2):
            if (row["spatial_prior"], row["temporal_prior"]) != (spatial_prior, temporal_prior):
                continue
            try:
                se, te = float(row["spatial_exponent"]), float(row["temporal_exponent"])
                n, total = int(row["iteration"]), int(row["iterations"])
                value = float(row["reconstructed_psnr_db"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value at CSV line {line}") from error
            if not np.isfinite([se, te]).all() or np.isnan(value) or np.isneginf(value):
                raise ValueError(f"Invalid exponent or PSNR at CSV line {line}")
            if not 1 <= n <= total:
                raise ValueError(f"Invalid iteration at CSV line {line}")
            run = runs.setdefault((se, te), {})
            if n in run:
                raise ValueError(f"Duplicate lambda pair/iteration at CSV line {line}")
            run[n] = value
            totals.add(total)
            constraints.add(row.get("intensity_constraint", "unspecified"))
    if not runs:
        raise ValueError(f"No {spatial_prior}/{temporal_prior} runs in this CSV")
    if len(totals) != 1 or len(constraints) != 1:
        raise ValueError("CSV mixes run lengths or intensity-constraint settings")
    total = totals.pop()
    if any(len(run) != total for run in runs.values()):
        raise ValueError("Incomplete iteration history; use a checkpoint containing completed runs")
    xs = sorted({se for se, _ in runs})
    ys = sorted({te for _, te in runs})
    values = np.full((len(ys), len(xs), total), np.nan)
    for (se, te), run in runs.items():
        values[ys.index(te), xs.index(se)] = [run[n] for n in range(1, total + 1)]
    return Sweep(xs, ys, np.arange(1, total + 1), values,
                 (spatial_prior, temporal_prior), constraints.pop())


class SweepViewer:
    def __init__(self, data, iteration=None):
        # Import pyplot only after the CLI selects its backend. In particular,
        # don't import main.py or plot_sweep_heatmaps.py: they select Agg.
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Slider

        self.data = data
        n = len(data.iterations)
        iteration = n if iteration is None else iteration
        if not 1 <= iteration <= n:
            raise ValueError(f"Iteration must be between 1 and {n}")
        self.index = iteration - 1
        finite = data.psnr[np.isfinite(data.psnr)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if low == high:
            low, high = low - .5, high + .5
        self.norm = Normalize(low, high)
        self.infinity_height = high + .1 * (high - low)
        self.figure, (self.heat_axis, self.curve_axis) = plt.subplots(1, 2, figsize=(13, 6))
        self.figure.subplots_adjust(left=.07, right=.97, bottom=.27, top=.83, wspace=.35)
        count = np.count_nonzero(~np.isnan(data.psnr[:, :, 0]))
        self.figure.suptitle(
            f"Spatial {data.pair[0].upper()} / temporal {data.pair[1].upper()} · "
            f"{count} loaded runs · intensity constraint: {data.constraint}", fontsize=14)
        cmap = plt.get_cmap("viridis").with_extremes(bad="#dddddd", over="#ff66cc")
        self.image = self.heat_axis.imshow(self._heat_values(), origin="lower", aspect="equal",
                                          cmap=cmap, norm=self.norm)
        self.figure.colorbar(self.image, ax=self.heat_axis, label="PSNR (dB)",
                             extend="max" if np.isposinf(data.psnr).any() else "neither")
        self.heat_axis.set_xticks(range(len(data.spatial)), [f"{v:g}" for v in data.spatial])
        self.heat_axis.set_yticks(range(len(data.temporal)), [f"{v:g}" for v in data.temporal])
        self.heat_axis.set_xlabel("Spatial λ exponent (λ = 10^exponent)")
        self.heat_axis.set_ylabel("Temporal λ exponent (λ = 10^exponent)")
        self.labels = {(y, x): self.heat_axis.text(x, y, "", ha="center", va="center", fontsize=9)
                       for y, x in np.ndindex(data.psnr.shape[:2])}
        self.selection = Rectangle((-.5, -.5), 1, 1, fill=False, edgecolor="#e85d04", linewidth=3)
        self.heat_axis.add_patch(self.selection)
        self.curve_axis.set_xlabel("Iteration")
        self.curve_axis.set_ylabel("PSNR (dB)")
        self.curve_axis.set_xlim(max(0, 1 - .02 * n), n + max(1, .02 * n))
        self.curve_axis.set_ylim(low - .05 * (high - low), self.infinity_height + .05 * (high - low))
        self.curve_axis.grid(alpha=.25)
        self.line, = self.curve_axis.plot([], [], color="#2457a7", linewidth=1.7)
        self.peak, = self.curve_axis.plot([], [], "*", color="#e85d04", markersize=12, label="Peak")
        self.final, = self.curve_axis.plot([], [], "s", color="#16855a", markersize=7, label="Final")
        self.current, = self.curve_axis.plot([], [], "o", color="black", markersize=5, label="Selected iteration")
        self.cursor = self.curve_axis.axvline(iteration, color="black", alpha=.35, linestyle="--")
        self.curve_axis.legend(loc="lower right", fontsize=8)
        self.detail = self.figure.text(.55, .16, "", fontsize=10, va="top")
        self.figure.text(.07, .17, "Click a heatmap cell to select its curve.\n"
                         "Colors stay fixed across all loaded iterations.\n"
                         "Gray = missing run; pink / ∞ = exact match.", fontsize=9, va="top")
        if np.isposinf(data.psnr).any():
            self.curve_axis.text(.02, .97, "∞ values shown above finite range", va="top",
                                  transform=self.curve_axis.transAxes, fontsize=8)
        slider_axis = self.figure.add_axes((.15, .055, .7, .035))
        self.slider = Slider(slider_axis, "Iteration", 1, max(n, 2),
                              valinit=iteration, valstep=1, valfmt="%d")
        if n == 1:
            self.slider.set_active(False)
        self.slider.on_changed(self.set_iteration)
        self.click_id = self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        # Initially select the best final PSNR among loaded runs.
        self.selected = np.unravel_index(np.nanargmax(data.psnr[:, :, -1]), data.psnr.shape[:2])
        self.select(*self.selected)
        self.set_iteration(iteration)

    def _heat_values(self):
        values = self.data.psnr[:, :, self.index]
        return np.ma.masked_invalid(np.where(np.isposinf(values), self.infinity_height, values))

    def _curve_values(self, values):
        return np.where(np.isposinf(values), self.infinity_height, values)

    def select(self, y, x):
        if np.isnan(self.data.psnr[y, x, 0]):
            return
        self.selected = (y, x)
        self.selection.set_xy((x - .5, y - .5))
        values = self.data.psnr[y, x]
        plotted = self._curve_values(values)
        best = int(np.argmax(values))
        self.line.set_data(self.data.iterations, plotted)
        self.peak.set_data([best + 1], [plotted[best]])
        self.final.set_data([len(values)], [plotted[-1]])
        se, te = self.data.spatial[x], self.data.temporal[y]
        self.curve_axis.set_title(f"Spatial λ = 10$^{{{se:g}}}$ · temporal λ = 10$^{{{te:g}}}$")
        self._update_current()

    def _update_current(self):
        values = self.data.psnr[self.selected]
        best = int(np.argmax(values))
        self.current.set_data([self.index + 1], [self._curve_values(values[self.index])])
        def score(value):
            return "∞" if np.isposinf(value) else f"{value:.4f}"
        self.detail.set_text(f"Selected: {score(values[self.index])} dB at iteration {self.index + 1}\n"
                             f"Peak: {score(values[best])} dB at iteration {best + 1}\n"
                             f"Final: {score(values[-1])} dB at iteration {len(values)}")
        self.figure.canvas.draw_idle()

    def set_iteration(self, value):
        self.index = int(np.clip(round(value), 1, len(self.data.iterations))) - 1
        self.image.set_data(self._heat_values())
        self.heat_axis.set_title(f"PSNR at iteration {self.index + 1}")
        for (y, x), label in self.labels.items():
            value = self.data.psnr[y, x, self.index]
            label.set_text("—" if np.isnan(value) else "∞" if np.isposinf(value) else f"{value:.2f}")
            label.set_color("white" if np.isfinite(value) and self.norm(value) < .5 else "black")
        self.cursor.set_xdata([self.index + 1, self.index + 1])
        self._update_current()

    def on_click(self, event):
        if event.inaxes is not self.heat_axis or event.button != 1:
            return
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(np.floor(event.xdata + .5)), int(np.floor(event.ydata + .5))
        if 0 <= x < len(self.data.spatial) and 0 <= y < len(self.data.temporal):
            self.select(y, x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?",
                        default=HERE / "results/smol_noise_sweep2/sweep_iterations.csv")
    parser.add_argument("--spatial-prior", choices=("tv", "wavelet"), default="tv")
    parser.add_argument("--temporal-prior", choices=("tv", "wavelet"), default="tv")
    parser.add_argument("--iteration", type=int, help="Initially selected iteration (default: final)")
    parser.add_argument("--save", type=Path, help="Save a static snapshot instead of opening a window")
    args = parser.parse_args()
    try:
        data = load_sweep(args.csv, args.spatial_prior, args.temporal_prior)
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        else:
            # Select the explicitly installed binding, independent of other Qt packages.
            from PySide6 import QtWidgets  # noqa: F401
            matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        viewer = SweepViewer(data, args.iteration)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            viewer.figure.savefig(args.save, dpi=160)
            plt.close(viewer.figure)
            print(f"Saved: {args.save}")
        else:
            print("Drag the iteration slider; click a heatmap cell to select a lambda pair.")
            print("This is a snapshot of completed runs. Reopen to load newer sweep checkpoints.")
            plt.show()
    except ImportError as error:
        parser.error(f"{error}. Install this directory's requirements.txt in your active environment.")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
