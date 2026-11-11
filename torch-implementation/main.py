"""Reconstruct an entire video jointly with FISTA and two selectable priors."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .inverse_solver import InverseSolver, adjoint_test
    from .video_io import load_psf, read_video, write_video
except ImportError:
    from inverse_solver import InverseSolver, adjoint_test
    from video_io import load_psf, read_video, write_video


def load_ground_truth(path, shape, fps):
    """Load aligned ground truth; only MP4 spatial downsampling is automatic."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        reference = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as dataset:
            if "reference" not in dataset:
                raise ValueError("Ground-truth NPZ must contain a 'reference' array")
            reference = dataset["reference"]
            if "fps" in dataset and not np.isclose(float(dataset["fps"]), fps, rtol=1e-4, atol=1e-3):
                raise ValueError("Ground-truth and measurement FPS differ")
    elif suffix == ".mp4":
        import imageio.v2 as imageio
        from PIL import Image

        frames = []
        with imageio.get_reader(str(path), format="FFMPEG") as reader:
            if not np.isclose(float(reader.get_meta_data()["fps"]), fps, rtol=1e-4, atol=1e-3):
                raise ValueError("Ground-truth and measurement FPS differ")
            source_shape = None
            for frame in reader:
                if len(frames) >= shape[3]:
                    raise ValueError("Ground-truth and measurement frame counts differ")
                if source_shape is None:
                    source_shape = frame.shape
                if frame.shape != source_shape:
                    raise ValueError("Ground-truth frames must have consistent dimensions")
                if frame.shape[0] < shape[0] or frame.shape[1] < shape[1]:
                    raise ValueError("Ground-truth MP4 cannot be smaller than the measurement")
                if frame.shape[:2] != tuple(shape[:2]):
                    frame = np.asarray(Image.fromarray(frame).resize(
                        (shape[1], shape[0]), Image.Resampling.LANCZOS))
                frames.append(frame.astype(np.float64) / 255.0)
        if len(frames) != shape[3]:
            raise ValueError("Ground-truth and measurement frame counts differ")
        reference = np.stack(frames, axis=-1)
    else:
        raise ValueError("--gt must be an .mp4, .npy, or .npz containing 'reference'")
    validate_reference(reference, shape)
    return reference


def validate_reference(reference, shape):
    if reference.shape != tuple(shape):
        raise ValueError(f"Ground-truth shape {reference.shape} differs from measurement {tuple(shape)}; "
                         "arrays must use (H,W,C,T)")
    if not np.issubdtype(reference.dtype, np.number) or np.iscomplexobj(reference) or not np.isfinite(reference).all():
        raise ValueError("Ground truth must contain finite real values")
    if np.any(reference < 0) or np.any(reference > 1):
        raise ValueError("Ground truth must be normalized to [0,1] for PSNR (data range 1)")


def quality_metrics(estimate, reference):
    """Pooled full-video MSE and PSNR, with peak intensity 1 and no clipping."""
    mse = float(np.mean((np.asarray(estimate, dtype=np.float64)
                         - np.asarray(reference, dtype=np.float64)) ** 2))
    # A string preserves exact-match infinity in standards-compliant JSON and
    # CSV; float(value) recovers infinity for numerical consumers.
    psnr = "Infinity" if mse == 0 else float(-10 * np.log10(mse))
    return mse, psnr


def write_csv(path, records):
    """Export all record fields, retaining numeric precision and CSV quoting."""
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()})


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
                       spatial_prior, temporal_prior, output_dir, *, save_artifacts=True):
    """Run one configuration and return its loss history and metric records."""
    solver = InverseSolver(psf, num_iters=args.num_iters,
                           lam_spatial=args.lambda_spatial, lam_temporal=args.lambda_temporal,
                           spatial_prior=spatial_prior, temporal_prior=temporal_prior,
                           wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                           prox_iters=args.prox_iters, prox_tol=args.prox_tol,
                           device=args.device, dtype=args.dtype)
    print(f"Device: {solver.device}; precision: {solver.dtype}", flush=True)
    # A small probe checks the operator without allocating another full video.
    lhs, rhs = adjoint_test(solver, measurement.shape[:2] + (1, 1))
    adjoint_tol = 1e-10 if solver.dtype == torch.float64 else 2e-5
    if not np.isclose(lhs, rhs, rtol=adjoint_tol, atol=adjoint_tol):
        raise RuntimeError(f"Adjoint test failed: {lhs} != {rhs}")

    reference_tensor = (torch.as_tensor(reference, device=solver.device, dtype=solver.dtype)
                        if reference is not None else None)
    quality_history = []

    def progress(iteration, loss, estimate):
        mse, psnr = None, None
        if reference_tensor is not None:
            # Evaluate on device; only one scalar crosses to the host. Never
            # retain iterates or transfer the full video at each iteration.
            mse = (estimate - reference_tensor).square().mean().item()
            psnr = "Infinity" if mse == 0 else float(-10 * np.log10(mse))
        quality_history.append((mse, psnr))
        if iteration == 1 or iteration % 10 == 0 or iteration == args.num_iters:
            quality = f", PSNR {float(psnr):.6g} dB" if psnr is not None else ""
            print(f"[{spatial_prior}/{temporal_prior}] FISTA {iteration}/{args.num_iters}: data loss {loss:.6g}{quality}", flush=True)

    result = solver(measurement, iterate_callback=progress).cpu().numpy()
    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "reconstructed.npy", result)
        np.save(output_dir / "data_losses.npy", solver.losses)
        np.save(output_dir / "psnr_history.npy",
                [float(psnr) if psnr is not None else np.nan for _, psnr in quality_history])
        write_video(output_dir / "reconstructed.mp4", result, fps)
    report = dict(input=str(args.input.resolve()), shape=list(result.shape), fps=fps,
                  spatial_prior=spatial_prior, temporal_prior=temporal_prior,
                  lambda_spatial=args.lambda_spatial, lambda_temporal=args.lambda_temporal,
                  iterations=args.num_iters, wavelet=args.wavelet, wavelet_level=args.wavelet_level,
                  prox_iters=args.prox_iters, prox_tol=args.prox_tol,
                  device=str(solver.device), dtype=str(solver.dtype),
                  torch_version=torch.__version__,
                  ground_truth=args.ground_truth_source,
                  psnr_data_range=1.0 if reference is not None else None,
                  final_data_loss=solver.losses[-1],
                  unconverged_prox_steps=sum(not d["converged"] for d in solver.prox_diagnostics),
                  blurred_mse=None, reconstructed_mse=None,
                  blurred_psnr_db=None, reconstructed_psnr_db=None)
    if reference is not None:
        if reference.shape != result.shape:
            raise ValueError("Reference and measurement dimensions differ")
        report["blurred_mse"], report["blurred_psnr_db"] = quality_metrics(measurement, reference)
        report["reconstructed_mse"], report["reconstructed_psnr_db"] = quality_metrics(result, reference)
        print(f"MSE: blurred={report['blurred_mse']:.6g}, reconstructed={report['reconstructed_mse']:.6g}")
        print(f"PSNR: blurred={float(report['blurred_psnr_db']):.6g} dB, "
              f"reconstructed={float(report['reconstructed_psnr_db']):.6g} dB")
    if save_artifacts:
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved reconstruction: {output_dir / 'reconstructed.mp4'}")
    iterations = [dict(spatial_prior=spatial_prior, temporal_prior=temporal_prior,
                       iteration=index, data_loss=loss,
                       lambda_spatial=args.lambda_spatial, lambda_temporal=args.lambda_temporal,
                       reconstructed_mse=quality_history[index - 1][0],
                       reconstructed_psnr_db=quality_history[index - 1][1],
                       prox_iterations=info["iterations"], prox_gap=info["gap"],
                       prox_converged=info["converged"])
                  for index, (loss, info) in enumerate(zip(solver.losses, solver.prox_diagnostics), start=1)]
    if save_artifacts:
        write_csv(output_dir / "iteration_metrics.csv", iterations)
    return np.asarray(solver.losses), report, iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="prep.py dataset.npz, or a real blurred MP4")
    parser.add_argument("--gt", type=Path,
                        help="Ground truth: MP4, (H,W,C,T) NPY, or NPZ with reference; overrides stored reference")
    parser.add_argument("--device", default="auto",
                        help="auto (CUDA, then MPS, then CPU), cuda[:index], mps, or cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto",
                        help="auto: float32 on GPU, float64 on CPU; MPS requires float32")
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
    args.ground_truth_source = None
    try:
        if args.gt is not None:
            reference = load_ground_truth(args.gt, measurement.shape, fps)
            args.ground_truth_source = str(args.gt.resolve())
        elif reference is not None:
            validate_reference(reference, measurement.shape)
            args.ground_truth_source = f"{args.input.resolve()}::reference"
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if reference is None:
        print("No ground truth available; MSE and PSNR fields will be empty.")
    pairs = ([(spatial, temporal) for spatial in ("wavelet", "tv")
              for temporal in ("wavelet", "tv")]
             if args.all_priors else [(args.spatial_prior, args.temporal_prior)])
    curves, reports, iterations = {}, [], []
    for spatial, temporal in pairs:
        output_dir = (args.output_dir / f"spatial_{spatial}_temporal_{temporal}"
                      if args.all_priors else args.output_dir)
        losses, report, history = run_reconstruction(
            args, measurement, psf, fps, reference, spatial, temporal, output_dir)
        curves[(spatial, temporal)] = losses
        reports.append(report)
        iterations.extend(history)
    if args.all_priors:
        import matplotlib.pyplot as plt

        write_csv(args.output_dir / "loss_all_priors.csv", iterations)
        write_csv(args.output_dir / "metrics_all_priors.csv", reports)
        print(f"Saved CSV histories and metrics: {args.output_dir}")
        figure = plot_prior_losses(curves)
        plot_path = args.output_dir / "loss_all_priors.png"
        figure.savefig(plot_path, dpi=160)
        print(f"Saved prior comparison: {plot_path}")
        if args.show_plot:
            plt.show()
        plt.close(figure)



if __name__ == "__main__":
    main()
