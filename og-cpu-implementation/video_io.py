"""Shared video IO and PSF scaling for preparation and reconstruction."""

from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform, gaussian_filter


def read_video(path, resolution=None, max_frames=None):
    """Return float64 RGB (H,W,C,T) in [0,1], fps, and original (W,H).

    resolution is (width, height), each strictly smaller than the source.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        fps = float(reader.get_meta_data()["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video must have a positive, finite frame rate")
        frames, source_size = [], None
        for index, frame in enumerate(reader):
            if max_frames is not None and index >= max_frames:
                break
            if source_size is None:
                source_size = (frame.shape[1], frame.shape[0])
                if resolution is not None and any(
                    target <= 0 or target >= source
                    for target, source in zip(resolution, source_size)
                ):
                    raise ValueError("Target width AND height must be positive and smaller than input")
            if (frame.shape[1], frame.shape[0]) != source_size:
                raise ValueError("All frames must have the same dimensions")
            if resolution is not None:
                frame = np.asarray(Image.fromarray(frame).resize(
                    tuple(resolution), Image.Resampling.LANCZOS))
            frames.append(frame.astype(np.float64) / 255.0)
    if not frames:
        raise ValueError("Video contains no frames")
    return np.stack(frames, axis=-1), fps, source_size


def write_video(path, video, fps):
    """Write a clipped 8-bit H.264 preview without resizing or per-frame gain.

    yuv444p and macro_block_size=1 preserve odd dimensions. Exact scientific
    arrays must be saved separately: this MP4 is lossy and contains no audio.
    """
    video = np.asarray(video)
    if video.ndim != 4 or video.shape[2] not in (1, 3) or not all(video.shape):
        raise ValueError("Expected (H,W,1|3,T) video")
    if not np.isfinite(video).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video and fps must be finite, with fps > 0")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), format="FFMPEG", fps=fps,
                            codec="libx264", pixelformat="yuv444p",
                            macro_block_size=1, quality=8) as writer:
        for index in range(video.shape[3]):
            frame = np.rint(np.clip(video[..., index], 0, 1) * 255).astype(np.uint8)
            if frame.shape[2] == 1:
                frame = np.repeat(frame, 3, axis=2)
            writer.append_data(frame)


def load_psf(path, source_size, target_size, psf_scale=1.0):
    """Make one centered, unit-sum 2D kernel shared by every RGB channel.

    psf_scale=1 means one TIFF pixel spans one ORIGINAL video pixel.
    Effective (x,y) scaling = psf_scale * target_size/source_size.
    RGB TIFFs are averaged across RGB (alpha ignored). The optical origin is
    assumed at (height//2,width//2); off-center PSFs should be centered first.
    """
    if not np.isfinite(psf_scale) or psf_scale <= 0:
        raise ValueError("psf_scale must be positive and finite")
    if len(source_size) != 2 or len(target_size) != 2 or any(
        not np.isfinite(v) or v <= 0 for v in (*source_size, *target_size)
    ):
        raise ValueError("Source and target sizes must contain positive width and height")
    psf = np.asarray(iio.imread(path), dtype=float)
    if psf.ndim == 3 and psf.shape[-1] in (1, 3, 4):
        psf = psf[..., :3].mean(axis=-1)
    if psf.ndim != 2 or not all(psf.shape) or not np.isfinite(psf).all():
        raise ValueError("TIFF must contain a finite 2D grayscale or RGB PSF")
    if np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("PSF must be nonnegative with positive total intensity")
    factors = psf_scale * np.asarray(target_size[::-1]) / np.asarray(source_size[::-1])
    if np.allclose(factors, 1, rtol=0, atol=1e-12):
        return psf / psf.sum()
    center = np.asarray(psf.shape) // 2
    # Anti-alias before shrinking. Resampling fixes the optical origin at the
    # output center even when dimensions change parity.
    sigma = np.maximum((1 / factors - 1) / 2, 0)
    filtered = gaussian_filter(psf, sigma=sigma, mode="constant")
    radius = np.ceil(np.maximum(center, np.asarray(psf.shape) - 1 - center) * factors).astype(int)
    scaled = affine_transform(filtered, np.diag(1 / factors),
                              offset=center - radius / factors,
                              output_shape=tuple(2 * radius + 1),
                              order=1, mode="constant", prefilter=False)
    if scaled.sum() <= 0:
        raise ValueError("Scaled PSF has zero energy; use a larger psf_scale")
    return scaled / scaled.sum()
