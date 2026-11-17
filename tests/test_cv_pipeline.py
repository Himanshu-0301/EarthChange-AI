"""Tests for the computer vision pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from earthchange.cv.pipeline import (
    REGION_SLICES,
    ChangeDetectionConfig,
    ChangeDetectionResult,
    InputValidationError,
    RegionStats,
    analyze_all_regions,
    analyze_region,
    area_calculation,
    detect_changes,
    generate_visualization,
    preprocess_images,
    validate_images,
)


# =====================================================================
# validate_images tests
# =====================================================================


def test_validate_images_valid_inputs() -> None:
    """Valid 3D arrays with identical shapes and finite values pass validation."""
    image_a = np.zeros((3, 10, 10), dtype=np.float32)
    image_b = np.ones((3, 10, 10), dtype=np.float32)

    assert validate_images(image_a, image_b) is True


def test_validate_images_shape_mismatch() -> None:
    """Mismatched shapes between image_a and image_b raise InputValidationError."""
    image_a = np.zeros((3, 10, 10), dtype=np.float32)
    image_b = np.zeros((3, 10, 12), dtype=np.float32)

    with pytest.raises(InputValidationError, match="shapes must match"):
        validate_images(image_a, image_b)


def test_validate_images_rejects_non_3d() -> None:
    """Non-3D inputs (e.g. 2D or 4D) raise InputValidationError."""
    image_2d = np.zeros((10, 10), dtype=np.float32)
    image_3d = np.zeros((1, 10, 10), dtype=np.float32)

    with pytest.raises(InputValidationError, match="3D numpy arrays"):
        validate_images(image_2d, image_3d)

    with pytest.raises(InputValidationError, match="3D numpy arrays"):
        validate_images(image_3d, np.zeros((1, 1, 10, 10), dtype=np.float32))


def test_validate_images_rejects_non_array() -> None:
    """Non-ndarray inputs raise InputValidationError."""
    with pytest.raises(InputValidationError, match="must be numpy ndarrays"):
        validate_images([[1, 2], [3, 4]], np.zeros((1, 2, 2)))  # type: ignore[arg-type]


def test_validate_images_rejects_empty_dimensions() -> None:
    """Images with zero in any dimension raise InputValidationError."""
    with pytest.raises(InputValidationError, match="non-empty spatial dimensions"):
        validate_images(np.zeros((0, 10, 10)), np.zeros((0, 10, 10)))

    with pytest.raises(InputValidationError, match="non-empty spatial dimensions"):
        validate_images(np.zeros((3, 0, 10)), np.zeros((3, 0, 10)))

    with pytest.raises(InputValidationError, match="non-empty spatial dimensions"):
        validate_images(np.zeros((3, 10, 0)), np.zeros((3, 10, 0)))


def test_validate_images_rejects_non_finite() -> None:
    """Images containing NaN or Inf raise InputValidationError."""
    image_a = np.zeros((2, 4, 4), dtype=np.float32)
    image_b = image_a.copy()
    image_a[0, 0, 0] = np.nan

    with pytest.raises(InputValidationError, match="finite values"):
        validate_images(image_a, image_b)

    image_a[0, 0, 0] = np.inf
    with pytest.raises(InputValidationError, match="finite values"):
        validate_images(image_a, image_b)


def test_validate_images_rejects_non_numeric() -> None:
    """Arrays with non-numeric dtype raise InputValidationError."""
    image_str = np.array([[["a"]]])
    with pytest.raises(InputValidationError, match="numeric data types"):
        validate_images(image_str, image_str)


# =====================================================================
# preprocess_images tests
# =====================================================================


def test_preprocess_images_converts_to_float32() -> None:
    """Integer arrays are converted to float32 without changing values."""
    image_a = np.array([[[0, 100], [200, 300]]], dtype=np.uint16)
    image_b = np.array([[[10, 110], [210, 310]]], dtype=np.int32)

    a_float, b_float = preprocess_images(image_a, image_b)

    assert a_float.dtype == np.float32
    assert b_float.dtype == np.float32
    assert np.allclose(a_float, image_a)
    assert np.allclose(b_float, image_b)


# =====================================================================
# detect_changes tests
# =====================================================================


def test_detect_changes_synthetic_default() -> None:
    """Detect changes on synthetic inputs using default Otsu threshold."""
    # Synthetic 2-band 4x4 image: change in 2x2 central region (4 pixels)
    image_a = np.zeros((2, 4, 4), dtype=np.float32)
    image_b = image_a.copy()
    image_b[:, 1:3, 1:3] = 10.0

    result = detect_changes(image_a, image_b)

    assert isinstance(result, ChangeDetectionResult)
    assert result.magnitude_map.shape == (4, 4)
    assert result.binary_mask.shape == (4, 4)
    assert result.binary_mask.dtype == np.bool_
    assert result.changed_pixels == 4
    assert result.total_pixels == 16
    assert result.changed_percentage == 25.0
    assert result.threshold_used > 0.0
    assert result.binary_mask[1:3, 1:3].all()
    assert not result.binary_mask[0, 0]


def test_detect_changes_manual_threshold() -> None:
    """Manual threshold overrides Otsu thresholding."""
    image_a = np.zeros((1, 4, 4), dtype=np.float32)
    image_b = image_a.copy()
    # Magnitude at changed pixels is 10.0
    image_b[0, 1:3, 1:3] = 10.0

    # Cutoff below 10 -> detects all 4 changed pixels
    config_low = ChangeDetectionConfig(threshold_method="manual", manual_threshold=5.0)
    result_low = detect_changes(image_a, image_b, config=config_low)
    assert result_low.threshold_used == 5.0
    assert result_low.changed_pixels == 4

    # Cutoff above 10 -> detects 0 pixels
    config_high = ChangeDetectionConfig(threshold_method="manual", manual_threshold=15.0)
    result_high = detect_changes(image_a, image_b, config=config_high)
    assert result_high.threshold_used == 15.0
    assert result_high.changed_pixels == 0


def test_detect_changes_morphology_removes_noise() -> None:
    """Morphological opening removes isolated 1x1 noise pixels."""
    # 8x8 synthetic image: 4x4 block of change + single isolated 1x1 noise pixel
    image_a = np.zeros((1, 8, 8), dtype=np.float32)
    image_b = image_a.copy()
    image_b[0, 1:5, 1:5] = 10.0  # 16 pixels
    image_b[0, 7, 7] = 10.0       # 1 isolated pixel

    # Without morphology
    config_no_morph = ChangeDetectionConfig(
        threshold_method="manual", manual_threshold=5.0, morphology_kernel=0
    )
    result_no_morph = detect_changes(image_a, image_b, config=config_no_morph)
    assert result_no_morph.changed_pixels == 17
    assert result_no_morph.binary_mask[7, 7]

    # With morphology opening and closing (kernel size 3)
    config_morph = ChangeDetectionConfig(
        threshold_method="manual", manual_threshold=5.0, morphology_kernel=3
    )
    result_morph = detect_changes(image_a, image_b, config=config_morph)
    # The isolated 1x1 pixel is removed by opening
    assert not result_morph.binary_mask[7, 7]
    # The 4x4 core remains
    assert result_morph.binary_mask[2:4, 2:4].all()
    assert result_morph.changed_pixels < result_no_morph.changed_pixels


def test_change_detection_config_validation() -> None:
    """Invalid configuration parameters raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported change detection method"):
        ChangeDetectionConfig(method="unsupported")

    with pytest.raises(ValueError, match="Unsupported threshold method"):
        ChangeDetectionConfig(threshold_method="invalid")

    with pytest.raises(ValueError, match="manual_threshold must be provided"):
        ChangeDetectionConfig(threshold_method="manual", manual_threshold=None)

    with pytest.raises(ValueError, match="morphology_kernel must be non-negative"):
        ChangeDetectionConfig(morphology_kernel=-1)


# =====================================================================
# analyze_region and REGION_SLICES tests
# =====================================================================


def test_analyze_region_named_regions() -> None:
    """Named regions return correct pixel counts and percentages."""
    # Known 8x8 mask where only upper-left 4x4 is True (16 pixels)
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True

    stats_full = analyze_region(mask, "full_image")
    assert stats_full.region_name == "full_image"
    assert stats_full.changed_pixels == 16
    assert stats_full.total_pixels == 64
    assert stats_full.changed_percentage == 25.0

    stats_ul = analyze_region(mask, "upper_left")
    assert stats_ul.changed_pixels == 16
    assert stats_ul.total_pixels == 16
    assert stats_ul.changed_percentage == 100.0

    stats_ur = analyze_region(mask, "upper_right")
    assert stats_ur.changed_pixels == 0
    assert stats_ur.total_pixels == 16
    assert stats_ur.changed_percentage == 0.0

    stats_ll = analyze_region(mask, "lower_left")
    assert stats_ll.changed_pixels == 0
    assert stats_ll.total_pixels == 16
    assert stats_ll.changed_percentage == 0.0

    stats_lr = analyze_region(mask, "lower_right")
    assert stats_lr.changed_pixels == 0
    assert stats_lr.total_pixels == 16
    assert stats_lr.changed_percentage == 0.0

    stats_top = analyze_region(mask, "top")
    assert stats_top.changed_pixels == 16
    assert stats_top.total_pixels == 32
    assert stats_top.changed_percentage == 50.0

    stats_bottom = analyze_region(mask, "bottom")
    assert stats_bottom.changed_pixels == 0
    assert stats_bottom.total_pixels == 32
    assert stats_bottom.changed_percentage == 0.0

    stats_left = analyze_region(mask, "left")
    assert stats_left.changed_pixels == 16
    assert stats_left.total_pixels == 32
    assert stats_left.changed_percentage == 50.0

    stats_right = analyze_region(mask, "right")
    assert stats_right.changed_pixels == 0
    assert stats_right.total_pixels == 32
    assert stats_right.changed_percentage == 0.0

    # Center is [2:6, 2:6] -> overlap with [0:4, 0:4] is [2:4, 2:4] (4 pixels)
    stats_center = analyze_region(mask, "center")
    assert stats_center.changed_pixels == 4
    assert stats_center.total_pixels == 16
    assert stats_center.changed_percentage == 25.0


def test_analyze_region_with_result_object() -> None:
    """analyze_region accepts a ChangeDetectionResult instance directly."""
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[0:2, 0:2] = True
    config = ChangeDetectionConfig()
    result = ChangeDetectionResult(
        magnitude_map=np.zeros((4, 4), dtype=np.float32),
        binary_mask=mask,
        threshold_used=1.0,
        changed_pixels=4,
        total_pixels=16,
        changed_percentage=25.0,
        config=config,
    )

    stats = analyze_region(result, "upper_left")
    assert stats.changed_pixels == 4
    assert stats.total_pixels == 4
    assert stats.changed_percentage == 100.0


def test_analyze_region_invalid_inputs() -> None:
    """Unknown region or non-2D mask raises ValueError."""
    mask = np.zeros((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="Unknown region"):
        analyze_region(mask, "nonexistent_region")

    with pytest.raises(ValueError, match="Mask must be a 2D array"):
        analyze_region(np.zeros((1, 4, 4), dtype=bool), "full_image")


def test_analyze_all_regions() -> None:
    """analyze_all_regions returns stats for all 10 predefined regions."""
    mask = np.zeros((8, 8), dtype=bool)
    all_stats = analyze_all_regions(mask)

    assert set(all_stats.keys()) == set(REGION_SLICES.keys())
    assert len(all_stats) == 10
    for name, stats in all_stats.items():
        assert isinstance(stats, RegionStats)
        assert stats.region_name == name
        assert stats.changed_pixels == 0
        assert stats.total_pixels > 0


# =====================================================================
# area_calculation tests
# =====================================================================


def test_area_calculation_without_resolution() -> None:
    """Without pixel resolution, area is None and pixel counts are preserved."""
    config = ChangeDetectionConfig()
    result = ChangeDetectionResult(
        magnitude_map=np.zeros((4, 4), dtype=np.float32),
        binary_mask=np.zeros((4, 4), dtype=np.bool_),
        threshold_used=1.0,
        changed_pixels=5,
        total_pixels=16,
        changed_percentage=31.25,
        config=config,
    )

    area_info = area_calculation(result)

    assert area_info == {
        "changed_pixels": 5,
        "total_pixels": 16,
        "changed_percentage": 31.25,
        "changed_area_m2": None,
        "pixel_resolution_m": None,
    }


def test_area_calculation_with_resolution() -> None:
    """With pixel resolution, area in square meters is computed accurately."""
    config = ChangeDetectionConfig()
    result = ChangeDetectionResult(
        magnitude_map=np.zeros((4, 4), dtype=np.float32),
        binary_mask=np.zeros((4, 4), dtype=np.bool_),
        threshold_used=1.0,
        changed_pixels=10,
        total_pixels=100,
        changed_percentage=10.0,
        config=config,
    )

    area_info = area_calculation(result, pixel_resolution=(10.0, 10.0))

    assert area_info["changed_pixels"] == 10
    assert area_info["total_pixels"] == 100
    assert area_info["changed_percentage"] == 10.0
    # 10 pixels * 10m * 10m = 1000 m2
    assert area_info["changed_area_m2"] == 1000.0
    assert area_info["pixel_resolution_m"] == (10.0, 10.0)


def test_area_calculation_invalid_resolution() -> None:
    """Non-positive or incorrectly shaped resolution raises ValueError."""
    config = ChangeDetectionConfig()
    result = ChangeDetectionResult(
        magnitude_map=np.zeros((4, 4), dtype=np.float32),
        binary_mask=np.zeros((4, 4), dtype=np.bool_),
        threshold_used=1.0,
        changed_pixels=0,
        total_pixels=16,
        changed_percentage=0.0,
        config=config,
    )

    with pytest.raises(ValueError, match="positive numbers"):
        area_calculation(result, pixel_resolution=(-10.0, 10.0))

    with pytest.raises(ValueError, match="positive numbers"):
        area_calculation(result, pixel_resolution=(10.0, 0.0))

    with pytest.raises(ValueError, match="2-tuple"):
        area_calculation(result, pixel_resolution=(10.0,))  # type: ignore[arg-type]


# =====================================================================
# generate_visualization tests
# =====================================================================


def test_generate_visualization_returns_figure() -> None:
    """generate_visualization returns a matplotlib Figure with proper subplots."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    image_a = np.zeros((3, 8, 8), dtype=np.float32)
    image_b = np.ones((3, 8, 8), dtype=np.float32)
    config = ChangeDetectionConfig()
    result = ChangeDetectionResult(
        magnitude_map=np.zeros((8, 8), dtype=np.float32),
        binary_mask=np.zeros((8, 8), dtype=np.bool_),
        threshold_used=0.5,
        changed_pixels=0,
        total_pixels=64,
        changed_percentage=0.0,
        config=config,
    )

    # Without ground truth: 4 subplots
    fig = generate_visualization(image_a, image_b, result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 4  # 4 image axes plus optional colorbar axis
    plt.close(fig)

    # With ground truth: 5 subplots
    gt = np.zeros((8, 8), dtype=bool)
    gt[2:4, 2:4] = True
    fig_gt = generate_visualization(image_a, image_b, result, ground_truth=gt)
    assert isinstance(fig_gt, Figure)
    assert len(fig_gt.axes) >= 5
    plt.close(fig_gt)


def test_generate_visualization_single_band() -> None:
    """generate_visualization handles single-band images."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    image_a = np.zeros((1, 8, 8), dtype=np.float32)
    image_b = np.ones((1, 8, 8), dtype=np.float32)
    config = ChangeDetectionConfig()
    result = ChangeDetectionResult(
        magnitude_map=np.zeros((8, 8), dtype=np.float32),
        binary_mask=np.zeros((8, 8), dtype=np.bool_),
        threshold_used=0.5,
        changed_pixels=0,
        total_pixels=64,
        changed_percentage=0.0,
        config=config,
    )

    fig = generate_visualization(image_a, image_b, result)
    assert isinstance(fig, Figure)
    plt.close(fig)
