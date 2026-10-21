"""Tests for loading OSCD image pairs from the documented directory layout."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from earthchange.cv.oscd_loader import OSCDDatasetError, load_oscd_pair


@pytest.fixture
def oscd_root(tmp_path: Path) -> Path:
    dataset = tmp_path / "Onera Satellite Change Detection dataset"
    images = dataset / "Onera Satellite Change Detection dataset - Images" / "beihai"
    transform = from_origin(100, 200, 10, 10)
    for time_index, value in ((1, 10), (2, 20)):
        directory = images / f"imgs_{time_index}_rect"
        directory.mkdir(parents=True)
        # Write in reverse creation order to verify deterministic band sorting.
        for band in ("B02", "B01"):
            with rasterio.open(
                directory / f"{band}.tif",
                "w",
                driver="GTiff",
                width=3,
                height=2,
                count=1,
                dtype="uint16",
                transform=transform,
            ) as raster:
                raster.write(np.full((2, 3), value + int(band[-1]), dtype=np.uint16), 1)
    (images / "dates.txt").write_text("2015-01-01\n2018-01-01\n", encoding="utf-8")

    mask_dir = dataset / "Onera Satellite Change Detection dataset - Train Labels" / "beihai" / "cm"
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8)).save(mask_dir / "cm.png")
    return tmp_path


def test_loads_pair_bands_dates_mask_and_metadata(oscd_root: Path) -> None:
    pair = load_oscd_pair(oscd_root, "beihai")

    assert pair.pair_id == "beihai"
    assert pair.image_a.shape == (2, 2, 3)
    assert pair.image_b.shape == (2, 2, 3)
    assert pair.image_a[:, 0, 0].tolist() == [11, 12]
    assert pair.image_b[:, 0, 0].tolist() == [21, 22]
    assert pair.change_mask.dtype == np.bool_
    assert pair.change_mask.tolist() == [[False, True, False], [True, False, False]]
    assert pair.dates == ("2015-01-01", "2018-01-01")
    assert pair.image_a_metadata.band_names == ("B01", "B02")
    assert pair.image_a_metadata.resolution == (10.0, 10.0)
    assert pair.image_a_metadata.width == 3
    assert pair.image_a_metadata.height == 2


def test_accepts_dataset_directory_itself(oscd_root: Path) -> None:
    dataset = oscd_root / "Onera Satellite Change Detection dataset"
    assert load_oscd_pair(dataset, "beihai").pair_id == "beihai"


def test_missing_pair_has_clear_error(oscd_root: Path) -> None:
    with pytest.raises(OSCDDatasetError, match="not found"):
        load_oscd_pair(oscd_root, "missing")


def test_rejects_pair_id_path_traversal(oscd_root: Path) -> None:
    with pytest.raises(OSCDDatasetError, match="single folder name"):
        load_oscd_pair(oscd_root, "../beihai")


def test_rejects_mask_with_incompatible_dimensions(oscd_root: Path) -> None:
    mask_path = next(oscd_root.glob("**/beihai/cm/cm.png"))
    Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(mask_path)

    with pytest.raises(OSCDDatasetError, match="does not match image shape"):
        load_oscd_pair(oscd_root, "beihai")
