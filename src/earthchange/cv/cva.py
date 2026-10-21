"""Experimental Change Vector Analysis (CVA) with Otsu thresholding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class CVAResult:
    """CVA magnitude, selected threshold, and binary change prediction."""

    magnitude: NDArray[np.float32]
    threshold: float
    change_mask: NDArray[np.bool_]


def cva_otsu(image_a: NDArray[np.generic], image_b: NDArray[np.generic]) -> CVAResult:
    """Compute per-pixel spectral change magnitude and threshold with Otsu.

    Inputs must use ``(bands, rows, columns)`` order and share the same shape.
    This is a simple experimental baseline; it does not register, normalize,
    or clean the images.
    """
    a = np.asarray(image_a)
    b = np.asarray(image_b)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Both images must have shape (bands, rows, columns)")
    if a.shape != b.shape:
        raise ValueError(f"Temporal image shapes must match; got {a.shape} and {b.shape}")
    if any(size == 0 for size in a.shape):
        raise ValueError("Images must have at least one band and non-empty spatial dimensions")

    a_float = a.astype(np.float32, copy=False)
    b_float = b.astype(np.float32, copy=False)
    if not np.isfinite(a_float).all() or not np.isfinite(b_float).all():
        raise ValueError("Images must contain only finite values")

    difference = b_float - a_float
    magnitude = np.sqrt(np.sum(difference * difference, axis=0, dtype=np.float32))
    threshold = float(threshold_otsu(magnitude))
    change_mask = magnitude > threshold
    return CVAResult(magnitude=magnitude, threshold=threshold, change_mask=change_mask)
