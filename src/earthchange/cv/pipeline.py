"""Computer vision pipeline for multi-temporal satellite change detection.

Provides image validation, preprocessing, Change Vector Analysis (CVA),
spatial region querying, area quantification, and visualization.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import opening

from earthchange.cv.cva import cva_otsu


class InputValidationError(ValueError):
    """Raised when CV input images fail validation requirements."""


def validate_images(image_a: NDArray[np.generic], image_b: NDArray[np.generic]) -> bool:
    """Validate that temporal images are compatible 3D finite numpy arrays.

    Parameters
    ----------
    image_a : NDArray
        First image, expected shape (bands, rows, cols).
    image_b : NDArray
        Second image, expected shape (bands, rows, cols).

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    InputValidationError
        If images are not 3D numpy arrays, have mismatched shapes, empty dimensions,
        non-numeric data types, or non-finite values.
    """
    if not isinstance(image_a, np.ndarray) or not isinstance(image_b, np.ndarray):
        raise InputValidationError("Both images must be numpy ndarrays")

    if image_a.ndim != 3 or image_b.ndim != 3:
        raise InputValidationError(
            f"Both images must be 3D numpy arrays with shape (bands, rows, cols); "
            f"got ndim={image_a.ndim} and ndim={image_b.ndim}"
        )

    if image_a.shape != image_b.shape:
        raise InputValidationError(
            f"Image shapes must match; got {image_a.shape} and {image_b.shape}"
        )

    if any(dim <= 0 for dim in image_a.shape):
        raise InputValidationError(
            "Images must have at least one band and non-empty spatial dimensions"
        )

    if not np.issubdtype(image_a.dtype, np.number) or not np.issubdtype(image_b.dtype, np.number):
        raise InputValidationError("Images must have numeric data types")

    if not np.isfinite(image_a).all() or not np.isfinite(image_b).all():
        raise InputValidationError("Images must contain only finite values")

    return True


def preprocess_images(
    image_a: NDArray[np.generic],
    image_b: NDArray[np.generic],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert input images to float32 without normalization.

    Parameters
    ----------
    image_a : NDArray
        First image, expected shape (bands, rows, cols).
    image_b : NDArray
        Second image, expected shape (bands, rows, cols).

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        Tuple containing both images cast to float32.
    """
    return (
        image_a.astype(np.float32, copy=False),
        image_b.astype(np.float32, copy=False),
    )


@dataclass(frozen=True)
class ChangeDetectionConfig:
    """Configuration settings for change detection.

    Attributes
    ----------
    method : str
        Algorithm name (currently only 'cva' is supported).
    threshold_method : str
        Thresholding strategy: 'otsu' or 'manual'.
    manual_threshold : float | None
        Cutoff threshold when threshold_method is 'manual'.
    morphology_kernel : int
        Kernel size for binary morphological cleanup (0 = disabled).
    """

    method: str = "cva"
    threshold_method: str = "otsu"
    manual_threshold: float | None = None
    morphology_kernel: int = 0

    def __post_init__(self) -> None:
        if self.method != "cva":
            raise ValueError(
                f"Unsupported change detection method: {self.method!r}. Supported: ['cva']"
            )
        if self.threshold_method not in ("otsu", "manual"):
            raise ValueError(
                f"Unsupported threshold method: {self.threshold_method!r}. Supported: ['otsu', 'manual']"
            )
        if self.threshold_method == "manual" and self.manual_threshold is None:
            raise ValueError("manual_threshold must be provided when threshold_method is 'manual'")
        if self.morphology_kernel < 0:
            raise ValueError("morphology_kernel must be non-negative")


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Outcome of change detection analysis.

    Attributes
    ----------
    magnitude_map : NDArray[np.float32]
        2D array of per-pixel change magnitude scores with shape (rows, cols).
    binary_mask : NDArray[np.bool_]
        2D boolean array where True indicates detected change.
    threshold_used : float
        Magnitude cutoff applied to produce the binary mask.
    changed_pixels : int
        Count of pixels flagged as changed.
    total_pixels : int
        Total number of pixels in the spatial domain.
    changed_percentage : float
        Percentage of changed pixels (0.0 to 100.0).
    config : ChangeDetectionConfig
        Configuration used to generate this result.
    """

    magnitude_map: NDArray[np.float32]
    binary_mask: NDArray[np.bool_]
    threshold_used: float
    changed_pixels: int
    total_pixels: int
    changed_percentage: float
    config: ChangeDetectionConfig


def detect_changes(
    image_a: NDArray[np.generic],
    image_b: NDArray[np.generic],
    config: ChangeDetectionConfig | None = None,
) -> ChangeDetectionResult:
    """Detect spectral changes between two temporal images.

    Validates inputs, preprocesses to float32, runs CVA with Otsu or manual
    thresholding, optionally cleans up noise via morphology opening and closing,
    and computes change statistics.

    Parameters
    ----------
    image_a : NDArray
        First temporal image of shape (bands, rows, cols).
    image_b : NDArray
        Second temporal image of shape (bands, rows, cols).
    config : ChangeDetectionConfig | None
        Configuration settings. If None, default configuration is used.

    Returns
    -------
    ChangeDetectionResult
        Detection results including magnitude map, binary mask, and summary statistics.
    """
    if config is None:
        config = ChangeDetectionConfig()

    validate_images(image_a, image_b)
    img_a_float, img_b_float = preprocess_images(image_a, image_b)

    cva_result = cva_otsu(img_a_float, img_b_float)
    magnitude = cva_result.magnitude

    if config.threshold_method == "manual":
        assert config.manual_threshold is not None
        threshold_used = float(config.manual_threshold)
        binary_mask = magnitude > threshold_used
    else:
        threshold_used = float(cva_result.threshold)
        binary_mask = cva_result.change_mask

    if config.morphology_kernel > 0:
        footprint = np.ones((config.morphology_kernel, config.morphology_kernel), dtype=bool)
        cleaned = opening(binary_mask, footprint=footprint)
        binary_mask = np.asarray(cleaned, dtype=np.bool_)
    else:
        binary_mask = np.asarray(binary_mask, dtype=np.bool_)

    total_pixels = int(binary_mask.size)
    changed_pixels = int(np.count_nonzero(binary_mask))
    changed_percentage = (
        float((changed_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0
    )

    return ChangeDetectionResult(
        magnitude_map=magnitude,
        binary_mask=binary_mask,
        threshold_used=threshold_used,
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        changed_percentage=changed_percentage,
        config=config,
    )


@dataclass(frozen=True)
class RegionStats:
    """Change statistics for a specific spatial region.

    Attributes
    ----------
    region_name : str
        Canonical name of the region.
    changed_pixels : int
        Count of changed pixels within this region.
    total_pixels : int
        Total number of pixels within this region.
    changed_percentage : float
        Percentage of changed pixels within this region (0.0 to 100.0).
    """

    region_name: str
    changed_pixels: int
    total_pixels: int
    changed_percentage: float


REGION_SLICES: dict[str, Callable[[int, int], tuple[slice, slice]]] = {
    "full_image": lambda h, w: (slice(0, h), slice(0, w)),
    "top": lambda h, w: (slice(0, h // 2), slice(0, w)),
    "bottom": lambda h, w: (slice(h // 2, h), slice(0, w)),
    "left": lambda h, w: (slice(0, h), slice(0, w // 2)),
    "right": lambda h, w: (slice(0, h), slice(w // 2, w)),
    "upper_left": lambda h, w: (slice(0, h // 2), slice(0, w // 2)),
    "upper_right": lambda h, w: (slice(0, h // 2), slice(w // 2, w)),
    "lower_left": lambda h, w: (slice(h // 2, h), slice(0, w // 2)),
    "lower_right": lambda h, w: (slice(h // 2, h), slice(w // 2, w)),
    "center": lambda h, w: (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4)),
}


def analyze_region(
    mask: NDArray[np.bool_] | ChangeDetectionResult,
    region_name: str,
) -> RegionStats:
    """Analyze change statistics within a named spatial region.

    Parameters
    ----------
    mask : NDArray[np.bool_] | ChangeDetectionResult
        2D boolean change mask or ChangeDetectionResult instance.
    region_name : str
        Name of the region (e.g. 'top', 'bottom', 'center', etc.).

    Returns
    -------
    RegionStats
        Change statistics for the specified region.

    Raises
    ------
    ValueError
        If region_name is not recognized or mask is not 2D.
    """
    if isinstance(mask, ChangeDetectionResult):
        raw_mask = mask.binary_mask
    else:
        raw_mask = np.asarray(mask)

    if raw_mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, got shape {raw_mask.shape}")

    if region_name not in REGION_SLICES:
        raise ValueError(
            f"Unknown region: {region_name!r}. Supported regions: {list(REGION_SLICES.keys())}"
        )

    h, w = raw_mask.shape
    row_slice, col_slice = REGION_SLICES[region_name](h, w)
    region_crop = raw_mask[row_slice, col_slice]

    total_pixels = int(region_crop.size)
    changed_pixels = int(np.count_nonzero(region_crop))
    changed_percentage = (
        float((changed_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0
    )

    return RegionStats(
        region_name=region_name,
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        changed_percentage=changed_percentage,
    )


def analyze_all_regions(
    mask: NDArray[np.bool_] | ChangeDetectionResult,
) -> dict[str, RegionStats]:
    """Analyze change statistics across all predefined spatial regions.

    Parameters
    ----------
    mask : NDArray[np.bool_] | ChangeDetectionResult
        2D boolean change mask or ChangeDetectionResult instance.

    Returns
    -------
    dict[str, RegionStats]
        Mapping from region name to its RegionStats.
    """
    return {name: analyze_region(mask, name) for name in REGION_SLICES}


def area_calculation(
    result: ChangeDetectionResult,
    pixel_resolution: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Calculate changed area and pixel statistics from change detection results.

    Parameters
    ----------
    result : ChangeDetectionResult
        Results from change detection.
    pixel_resolution : tuple[float, float] | None
        Spatial resolution in meters per pixel as (res_x, res_y).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - changed_pixels (int)
        - total_pixels (int)
        - changed_percentage (float)
        - changed_area_m2 (float | None)
        - pixel_resolution_m (tuple[float, float] | None)
    """
    if pixel_resolution is not None:
        if not isinstance(pixel_resolution, (tuple, list)) or len(pixel_resolution) != 2:
            raise ValueError("pixel_resolution must be a 2-tuple (res_x, res_y)")
        try:
            res_x, res_y = float(pixel_resolution[0]), float(pixel_resolution[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("pixel_resolution must contain finite positive numbers in metres") from exc
        if not math.isfinite(res_x) or not math.isfinite(res_y) or res_x <= 0 or res_y <= 0:
            raise ValueError("pixel_resolution must contain finite positive numbers in metres")
        changed_area_m2: float | None = float(result.changed_pixels * res_x * res_y)
        pixel_resolution_m: tuple[float, float] | None = (res_x, res_y)
    else:
        changed_area_m2 = None
        pixel_resolution_m = None

    return {
        "changed_pixels": result.changed_pixels,
        "total_pixels": result.total_pixels,
        "changed_percentage": result.changed_percentage,
        "changed_area_m2": changed_area_m2,
        "pixel_resolution_m": pixel_resolution_m,
    }


def _normalize_channel(channel: NDArray[np.generic]) -> NDArray[np.float32]:
    """Normalize a 2D image channel to [0, 1] for visualization display."""
    arr = channel.astype(np.float32, copy=False)
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr, dtype=np.float32)


def _prepare_display_image(img: NDArray[np.generic]) -> tuple[NDArray[np.float32], str | None]:
    """Prepare a 2D or 3D image array for display in matplotlib.

    Returns the normalized array and optional colormap name (None for RGB).
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return _normalize_channel(arr), "gray"
    if arr.ndim == 3:
        if arr.shape[0] >= 3:
            r = _normalize_channel(arr[0])
            g = _normalize_channel(arr[1])
            b = _normalize_channel(arr[2])
            return np.stack([r, g, b], axis=-1), None
        return _normalize_channel(arr[0]), "gray"
    raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")


def generate_visualization(
    image_a: NDArray[np.generic],
    image_b: NDArray[np.generic],
    result: ChangeDetectionResult,
    ground_truth: NDArray[np.generic] | None = None,
) -> Any:
    """Create a matplotlib figure displaying images, change magnitude, and masks.

    Subplots created:
    - Image A (band 0 or RGB composite of first 3 bands)
    - Image B (same)
    - Change magnitude heatmap
    - Binary change mask
    - Ground truth comparison overlay (if ground_truth provided)

    Parameters
    ----------
    image_a : NDArray
        First temporal image of shape (bands, rows, cols) or (rows, cols).
    image_b : NDArray
        Second temporal image of shape (bands, rows, cols) or (rows, cols).
    result : ChangeDetectionResult
        Detection result containing magnitude map and binary mask.
    ground_truth : NDArray | None
        Optional ground truth mask for comparison overlay.

    Returns
    -------
    matplotlib.figure.Figure
        Created figure with subplots. Caller may show or save as needed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for generate_visualization. "
            "Please install matplotlib to use this function."
        ) from exc

    has_gt = ground_truth is not None
    num_cols = 5 if has_gt else 4
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    if num_cols == 1:
        axes = np.array([axes])

    disp_a, cmap_a = _prepare_display_image(image_a)
    axes[0].imshow(disp_a, cmap=cmap_a)
    axes[0].set_title("Image A (Before)")
    axes[0].axis("off")

    disp_b, cmap_b = _prepare_display_image(image_b)
    axes[1].imshow(disp_b, cmap=cmap_b)
    axes[1].set_title("Image B (After)")
    axes[1].axis("off")

    im_mag = axes[2].imshow(result.magnitude_map, cmap="viridis")
    axes[2].set_title(f"Change Magnitude\n(thr={result.threshold_used:.2f})")
    axes[2].axis("off")
    fig.colorbar(im_mag, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(result.binary_mask, cmap="gray")
    axes[3].set_title(f"Binary Change Mask\n({result.changed_percentage:.1f}% changed)")
    axes[3].axis("off")

    if has_gt:
        gt = np.asarray(ground_truth)
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]
        gt_bool = gt > 0

        pred_bool = result.binary_mask
        overlay = np.zeros((*pred_bool.shape, 3), dtype=np.float32)
        # TP: Green, FP: Red, FN: Blue, TN: Dark gray
        tp = pred_bool & gt_bool
        fp = pred_bool & (~gt_bool)
        fn = (~pred_bool) & gt_bool
        tn = (~pred_bool) & (~gt_bool)

        overlay[tp] = [0.0, 0.8, 0.0]
        overlay[fp] = [0.9, 0.1, 0.1]
        overlay[fn] = [0.1, 0.3, 0.9]
        overlay[tn] = [0.15, 0.15, 0.15]

        axes[4].imshow(overlay)
        axes[4].set_title("Ground Truth Overlay\n(G:TP, R:FP, B:FN)")
        axes[4].axis("off")

    fig.tight_layout()
    return fig
