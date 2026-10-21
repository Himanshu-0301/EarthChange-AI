"""Tests for the experimental CVA baseline and its evaluation metrics."""

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from experiments.cv_feasibility import binary_metrics, run
from PIL import Image
from rasterio.transform import from_origin

from earthchange.cv.cva import cva_otsu


def test_cva_otsu_detects_pixels_with_spectral_change() -> None:
    image_a = np.zeros((2, 4, 4), dtype=np.uint16)
    image_b = image_a.copy()
    image_b[:, 1:3, 1:3] = 8

    result = cva_otsu(image_a, image_b)

    assert result.magnitude.shape == (4, 4)
    assert result.magnitude.dtype == np.float32
    assert result.change_mask.dtype == np.bool_
    assert result.change_mask.sum() == 4
    assert result.change_mask[1:3, 1:3].all()
    assert not result.change_mask[0, 0]


def test_cva_otsu_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        cva_otsu(np.zeros((2, 3, 3)), np.zeros((2, 4, 3)))


def test_cva_otsu_rejects_non_finite_values() -> None:
    image = np.zeros((1, 2, 2), dtype=np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        cva_otsu(image, np.zeros_like(image))


def test_binary_metrics_match_expected_pixel_counts() -> None:
    prediction = np.array([[True, True], [False, False]])
    reference = np.array([[True, False], [True, False]])

    assert binary_metrics(prediction, reference) == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "iou": 1 / 3,
    }


def test_binary_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        binary_metrics(np.zeros((2, 2)), np.zeros((2, 3)))


def test_feasibility_runner_writes_pair_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "Onera Satellite Change Detection dataset"
    image_root = dataset / "Onera Satellite Change Detection dataset - Images" / "sample"
    transform = from_origin(0, 40, 10, 10)
    expected_change = np.zeros((4, 4), dtype=bool)
    expected_change[1:3, 1:3] = True

    for time_index, value in ((1, 0), (2, 5)):
        directory = image_root / f"imgs_{time_index}_rect"
        directory.mkdir(parents=True)
        array = np.where(expected_change, value, 0).astype(np.uint16)
        with rasterio.open(
            directory / "B01.tif",
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="uint16",
            transform=transform,
        ) as raster:
            raster.write(array, 1)

    mask_dir = dataset / "Onera Satellite Change Detection dataset - Train Labels" / "sample" / "cm"
    mask_dir.mkdir(parents=True)
    Image.fromarray(expected_change.astype(np.uint8) * 255).save(mask_dir / "cm.png")
    output = tmp_path / "results" / "metrics.csv"

    rows = run(tmp_path, ["sample"], output)

    assert rows[0]["pair_id"] == "sample"
    assert rows[0]["f1"] == 1.0
    assert rows[0]["iou"] == 1.0
    with output.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["pair_id"] == "sample"
    assert float(csv_row["f1"]) == 1.0
