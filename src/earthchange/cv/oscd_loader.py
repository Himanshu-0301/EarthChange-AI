"""Load registered OSCD image pairs and their binary change masks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from PIL import Image


class OSCDDatasetError(ValueError):
    """Raised when an OSCD pair is missing or incompatible."""


@dataclass(frozen=True)
class RasterMetadata:
    """Spatial and band metadata recorded by a source raster."""

    width: int
    height: int
    band_count: int
    band_names: tuple[str, ...]
    resolution: tuple[float, float] | None
    crs: str | None


@dataclass(frozen=True)
class OSCDPair:
    """A loaded OSCD temporal pair and binary change mask.

    Images use ``(bands, rows, columns)`` order. Mask values are booleans,
    where ``True`` denotes change according to the OSCD label convention.
    """

    pair_id: str
    image_a: NDArray[np.generic]
    image_b: NDArray[np.generic]
    change_mask: NDArray[np.bool_]
    dates: tuple[str, str] | None
    image_a_metadata: RasterMetadata
    image_b_metadata: RasterMetadata


_BAND_PATTERN = re.compile(r"B(?:0[1-9]|1[0-2]|8A)$", re.IGNORECASE)


def _band_name(dataset: rasterio.DatasetReader, index: int) -> str:
    """Return a source-provided band name, or an honest positional label."""
    description = dataset.descriptions[index - 1]
    if description:
        return description
    return f"band_{index}"


def _metadata(dataset: rasterio.DatasetReader) -> RasterMetadata:
    transform = dataset.transform
    resolution = (abs(transform.a), abs(transform.e))
    if resolution == (0.0, 0.0):
        resolution = None
    return RasterMetadata(
        width=dataset.width,
        height=dataset.height,
        band_count=dataset.count,
        band_names=tuple(_band_name(dataset, i) for i in range(1, dataset.count + 1)),
        resolution=resolution,
        crs=str(dataset.crs) if dataset.crs else None,
    )


def _read_raster(path: Path) -> tuple[NDArray[np.generic], RasterMetadata]:
    try:
        with rasterio.open(path) as dataset:
            image = dataset.read()
            metadata = _metadata(dataset)
    except rasterio.errors.RasterioError as exc:
        raise OSCDDatasetError(f"Could not read raster {path}: {exc}") from exc
    if image.ndim != 3 or image.shape[0] == 0:
        raise OSCDDatasetError(f"Raster must contain at least one band: {path}")
    return image, metadata


def _read_mask(path: Path) -> NDArray[np.bool_]:
    try:
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"))
    except (OSError, ValueError) as exc:
        raise OSCDDatasetError(f"Could not read change mask {path}: {exc}") from exc
    # OSCD stores unchanged pixels as 0 and changed pixels as 255.
    return mask > 0


def _read_dates(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    dates = tuple(line.split()[-1] for line in path.read_text(encoding="utf-8").splitlines() if line.split())
    if len(dates) != 2:
        raise OSCDDatasetError(f"Expected exactly two dates in {path}, found {len(dates)}")
    return dates[0], dates[1]


def load_oscd_pair(dataset_root: str | Path, pair_id: str) -> OSCDPair:
    """Load a named pair from an extracted OSCD dataset root.

    The root may be the dataset's parent directory or the ``Onera Satellite
    Change Detection dataset`` directory itself. The standard OSCD layout is
    ``Images/<pair>/imgs_1_rect/*.tif``, ``imgs_2_rect/*.tif``,
    ``Images/<pair>/dates.txt`` and ``<Split> Labels/<pair>/cm/cm.png``.
    Split labels are discovered only when exactly one matching mask exists.
    """
    if not pair_id or Path(pair_id).name != pair_id or pair_id in {".", ".."}:
        raise OSCDDatasetError(f"pair_id must be a single folder name, got {pair_id!r}")

    root = Path(dataset_root).expanduser()
    dataset_dir = root / "Onera Satellite Change Detection dataset"
    if not dataset_dir.is_dir():
        dataset_dir = root
    image_root = dataset_dir / "Onera Satellite Change Detection dataset - Images"
    if not image_root.is_dir():
        raise OSCDDatasetError(f"OSCD image directory not found under {dataset_dir}")

    pair_dir = image_root / pair_id
    if not pair_dir.is_dir():
        raise OSCDDatasetError(f"OSCD pair {pair_id!r} not found in {image_root}")

    def image_paths(time_index: int) -> list[Path]:
        folder = pair_dir / f"imgs_{time_index}_rect"
        if not folder.is_dir():
            raise OSCDDatasetError(f"Missing temporal image directory: {folder}")
        paths = sorted(folder.glob("*.tif"))
        if not paths:
            raise OSCDDatasetError(f"No GeoTIFF bands found in {folder}")
        # OSCD often names individual band TIFFs by band ID. Sort them
        # deterministically in Sentinel-2 spectral order where recognized.
        known = {f"B{index:02d}": index for index in range(1, 13)}
        known["B8A"] = 8.5
        if all(_BAND_PATTERN.fullmatch(path.stem) for path in paths):
            paths.sort(key=lambda path: known[path.stem.upper()])
        return paths

    paths_a, paths_b = image_paths(1), image_paths(2)
    if len(paths_a) != len(paths_b):
        raise OSCDDatasetError(
            f"Temporal image band counts differ for {pair_id!r}: {len(paths_a)} vs {len(paths_b)}"
        )

    def load_bands(paths: list[Path]) -> tuple[NDArray[np.generic], RasterMetadata]:
        arrays: list[NDArray[np.generic]] = []
        first_metadata: RasterMetadata | None = None
        for path in paths:
            array, metadata = _read_raster(path)
            if metadata.band_count != 1:
                raise OSCDDatasetError(
                    f"Expected one band per OSCD GeoTIFF file, found {metadata.band_count}: {path}"
                )
            if first_metadata and (metadata.width, metadata.height) != (
                first_metadata.width,
                first_metadata.height,
            ):
                raise OSCDDatasetError(f"Incompatible band dimensions in temporal image: {path}")
            first_metadata = first_metadata or metadata
            arrays.append(array[0])
        assert first_metadata is not None
        return np.stack(arrays, axis=0), RasterMetadata(
            width=first_metadata.width,
            height=first_metadata.height,
            band_count=len(arrays),
            band_names=tuple(path.stem for path in paths),
            resolution=first_metadata.resolution,
            crs=first_metadata.crs,
        )

    image_a, metadata_a = load_bands(paths_a)
    image_b, metadata_b = load_bands(paths_b)
    if (metadata_a.width, metadata_a.height) != (metadata_b.width, metadata_b.height):
        raise OSCDDatasetError(f"Temporal image dimensions differ for pair {pair_id!r}")
    if metadata_a.band_names != metadata_b.band_names:
        raise OSCDDatasetError(f"Temporal image band names differ for pair {pair_id!r}")

    mask_matches = sorted(dataset_dir.glob(f"Onera Satellite Change Detection dataset - * Labels/{pair_id}/cm/cm.png"))
    if len(mask_matches) != 1:
        if not mask_matches:
            raise OSCDDatasetError(f"No ground-truth mask found for pair {pair_id!r} under {dataset_dir}")
        raise OSCDDatasetError(
            f"Multiple ground-truth masks found for pair {pair_id!r}; specify a unique dataset root"
        )
    change_mask = _read_mask(mask_matches[0])
    if change_mask.shape != (metadata_a.height, metadata_a.width):
        raise OSCDDatasetError(
            f"Change mask shape {change_mask.shape} does not match image shape "
            f"{(metadata_a.height, metadata_a.width)} for pair {pair_id!r}"
        )

    return OSCDPair(
        pair_id=pair_id,
        image_a=image_a,
        image_b=image_b,
        change_mask=change_mask,
        dates=_read_dates(pair_dir / "dates.txt"),
        image_a_metadata=metadata_a,
        image_b_metadata=metadata_b,
    )
