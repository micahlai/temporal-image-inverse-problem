"""Reconstruct an entire video jointly with FISTA and two selectable priors."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .inverse_solver import InverseSolver, adjoint_test
    from .video_io import load_psf, read_video, write_video
except ImportError:
    from inverse_solver import InverseSolver, adjoint_test
    from video_io import load_psf, read_video, write_video


def lambda_from_exponent(value):
    """Parse a base-10 exponent into a finite, positive solver weight."""
    try:
        exponent = float(value)
        weight = 10.0 ** exponent
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Expected a base-10 exponent, e.g. -5 for 1e-5") from error
    if not np.isfinite(exponent) or not np.isfinite(weight) or weight <= 0:
        raise argparse.ArgumentTypeError("Exponent must produce a finite, positive lambda")
    return weight


def boolean_argument(value):
    """Accept explicit true/false values without Python's bool('false') trap."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected true or false")


def plot_prior_losses(curves):
    """Create a common log-loss / linear-iteration comparison figure."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    has_positive_loss = False
    for (spatial, temporal), losses in curves.items():
        losses = np.asarray(losses, dtype=float)
        has_positive_loss |= bool(np.any(np.isfinite(losses) & (losses > 0)))
        # Logarithms are undefined at zero. Keep the original losses on disk;
        # mask zero/nonfinite values in the figure instead of inventing a floor.
        plotted = np.ma.masked_where(~np.isfinite(losses) | (losses <= 0), losses)
        axis.plot(np.arange(1, len(losses) + 1), plotted,
                  label=f"Spatial: {spatial}, temporal: {temporal}")
    if not has_positive_loss:
        axis.set_ylim(1e-16, 1)
    axis.set_yscale("log")
    axis.set_xscale("linear")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Data loss (log scale)")
    axis.set_title("FISTA: all spatial / temporal prior combinations")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    return figure


def run_reconstruction(args, measurement, psf, fps, reference,
                       spatial_prior, temporal_prior, output_dir):
    """Run one configuration, write its artifacts, and retain only its losses."""
    solver = InverseSolver(psf, num_iters=args.num_iters,
                           lam_spatial=args.lambda_spatial, lam_temporal=args.lambda_temporal,
                           spatial_prior=spatial_prior, temporal_prior=temporal_prior,
                           wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                           prox_iters=args.prox_iters, prox_tol=args.prox_tol)
    # A small probe checks the operator without allocating another full video.
    lhs, rhs = adjoint_test(solver, measurement.shape[:2] + (1, 1))
    if not np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10):
        raise RuntimeError(f"Adjoint test failed: {lhs} != {rhs}")

    def progress(iteration, loss):
        if iteration == 1 or iteration % 10 == 0 or iteration == args.num_iters:
            print(f"[{spatial_prior}/{temporal_prior}] FISTA {iteration}/{args.num_iters}: data loss {loss:.6g}", flush=True)

    result = solver(measurement, callback=progress)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "reconstructed.npy", result)
    np.save(output_dir / "data_losses.npy", solver.losses)
    write_video(output_dir / "reconstructed.mp4", result, fps)
    report = dict(input=str(args.input.resolve()), shape=list(result.shape), fps=fps,
                  spatial_prior=spatial_prior, temporal_prior=temporal_prior,
                  lambda_spatial=args.lambda_spatial, lambda_temporal=args.lambda_temporal,
                  iterations=args.num_iters, wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                  prox_iters=args.prox_iters, prox_tol=args.prox_tol,
                  final_data_loss=solver.losses[-1],
                  unconverged_prox_steps=sum(not d["converged"] for d in solver.prox_diagnostics))
    if reference is not None:
        if reference.shape != result.shape:
            raise ValueError("Reference and measurement dimensions differ")
        report["blurred_mse"] = float(np.mean((measurement - reference)**2))
        report["reconstructed_mse"] = float(np.mean((result - reference)**2))
        print(f"MSE: blurred={report['blurred_mse']:.6g}, reconstructed={report['reconstructed_mse']:.6g}")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved reconstruction: {output_dir / 'reconstructed.mp4'}")
    return np.asarray(solver.losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="prep.py dataset.npz, or a real blurred MP4")
    parser.add_argument("--psf", type=Path, help="TIFF PSF, required only for MP4 input")
    parser.add_argument("--psf-scale", type=float, default=1.0,
                        help="For MP4 input: PSF pixel size relative to input video pixels")
    parser.add_argument("--spatial-prior", choices=("wavelet", "tv"), default="wavelet")
    parser.add_argument("--temporal-prior", choices=("wavelet", "tv"), default="tv")
    parser.add_argument("--all-priors", type=boolean_argument, nargs="?", const=True,
                        default=False, metavar="true|false",
                        help="Run all four combinations; ignores --spatial-prior and --temporal-prior (default: false)")
    parser.add_argument("--show-plot", type=boolean_argument, default=True,
                        metavar="true|false",
                        help="Display the all-priors plot after saving it (default: true)")
    parser.add_argument("--lambda-spatial", type=lambda_from_exponent, default="-5",
                        metavar="EXPONENT", help="Base-10 exponent: -5 means 1e-5 (default: -5)")
    parser.add_argument("--lambda-temporal", type=lambda_from_exponent, default="-5",
                        metavar="EXPONENT", help="Base-10 exponent: -5 means 1e-5 (default: -5)")
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--prox-iters", type=int, default=200)
    parser.add_argument("--prox-tol", type=float, default=1e-5)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructed"))
    args = parser.parse_args()
    reference = None
    if args.input.suffix.lower() == ".npz":
        if args.psf is not None or args.psf_scale != 1.0:
            parser.error("Synthetic datasets already contain the exact scaled PSF; omit --psf/--psf-scale")
        with np.load(args.input, allow_pickle=False) as dataset:
            measurement, psf, fps = dataset["measurement"], dataset["psf"], float(dataset["fps"])
            if "reference" in dataset:
                reference = dataset["reference"]
    elif args.input.suffix.lower() == ".mp4":
        if args.psf is None:
            parser.error("--psf is required for an MP4 input")
        measurement, fps, source_size = read_video(args.input)
        psf = load_psf(args.psf, source_size, source_size, args.psf_scale)
    else:
        parser.error("Input must be a dataset.npz or an .mp4")
    if reference is not None and reference.shape != measurement.shape:
        raise ValueError("Reference and measurement dimensions differ")
    pairs = ([(spatial, temporal) for spatial in ("wavelet", "tv")
              for temporal in ("wavelet", "tv")]
             if args.all_priors else [(args.spatial_prior, args.temporal_prior)])
    curves = {}
    for spatial, temporal in pairs:
        output_dir = (args.output_dir / f"spatial_{spatial}_temporal_{temporal}"
                      if args.all_priors else args.output_dir)
        curves[(spatial, temporal)] = run_reconstruction(
            args, measurement, psf, fps, reference, spatial, temporal, output_dir)
    if args.all_priors:
        import matplotlib.pyplot as plt

        figure = plot_prior_losses(curves)
        plot_path = args.output_dir / "loss_all_priors.png"
        figure.savefig(plot_path, dpi=160)
        print(f"Saved prior comparison: {plot_path}")
        if args.show_plot:
            plt.show()
        plt.close(figure)



if __name__ == "__main__":
    main()
