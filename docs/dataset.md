# OSCD Dataset Specification & Data Loading

This document outlines the structure of the **Onera Satellite Change Detection (OSCD)** benchmark dataset, what the `earthchange.cv.oscd_loader` loader expects and provides, explicit assumptions, and what remains unknown.

---

## 1. Verified OSCD Dataset Structure

The Onera Satellite Change Detection (OSCD) dataset addresses binary change detection between multi-temporal Sentinel-2 satellite image pairs acquired over 24 global cities (14 training, 10 testing).

### Directory Hierarchy (Official Archives)

The official release archive (`Images.zip`, `Train Labels.zip`, `Test Labels.zip`) unpacks into the following layout:

```text
Onera Satellite Change Detection dataset/
├── Onera Satellite Change Detection dataset - Images/
│   ├── <pair_id>/                          # e.g., 'beirut', 'beihai', 'paris'
│   │   ├── imgs_1_rect/                    # Time 1 (pre-change) rectified bands
│   │   │   ├── B01.tif
│   │   │   ├── B02.tif
│   │   │   ├── ...
│   │   │   └── B12.tif
│   │   ├── imgs_2_rect/                    # Time 2 (post-change) rectified bands
│   │   │   ├── B01.tif
│   │   │   ├── B02.tif
│   │   │   ├── ...
│   │   │   └── B12.tif
│   │   └── dates.txt                       # Acquisition dates for Time 1 and Time 2
│   └── ...
├── Onera Satellite Change Detection dataset - Train Labels/
│   ├── <pair_id>/
│   │   └── cm/
│   │       └── cm.png                      # Ground-truth binary change mask
│   └── ...
└── Onera Satellite Change Detection dataset - Test Labels/
    ├── <pair_id>/
    │   └── cm/
    │       └── cm.png                      # Ground-truth binary change mask
    └── ...
```

---

## 2. Band Information & Resolutions

Sentinel-2 imagery comprises 13 spectral bands with different native spatial resolutions:

| Resolution | Bands | Description |
|---|---|---|
| **10m** | `B02` (Blue), `B03` (Green), `B04` (Red), `B08` (Broad NIR) | High-resolution visual & NIR bands |
| **20m** | `B05`, `B06`, `B07` (Red Edge), `B8A` (Narrow NIR), `B11`, `B12` (SWIR) | Vegetation red edge & short-wave infrared |
| **60m** | `B01` (Coastal aerosol), `B09` (Water vapour), `B10` (Cirrus) | Atmospheric characterization |

---

## 3. What the Loader Expects

[`load_oscd_pair(dataset_root, pair_id)`](file:///d:/Earth%20ChangeAi/src/earthchange/cv/oscd_loader.py) expects:

1. **Dataset Root:** The directory pointing either to the parent containing `Onera Satellite Change Detection dataset` or directly to `Onera Satellite Change Detection dataset`.
2. **Pair Identifier:** A sanitized directory name corresponding to an OSCD region (e.g. `beihai`, `beirut`). Path traversal or invalid identifiers are rejected.
3. **Temporal Image Bands:** `imgs_1_rect/*.tif` and `imgs_2_rect/*.tif` containing single-band GeoTIFFs. Bands are deterministically sorted in Sentinel-2 spectral order (`B01` through `B12`, including `B8A`).
4. **Change Mask:** Exactly one matching ground-truth mask `cm/cm.png` under either `Train Labels` or `Test Labels`.
5. **Dimensional Consistency:** Spatial dimensions of all loaded bands and the change mask must match.

---

## 4. What Is Loaded

The loader returns a typed [`OSCDPair`](file:///d:/Earth%20ChangeAi/src/earthchange/cv/oscd_loader.py) dataclass containing:

* `pair_id`: Region identifier string.
* `image_a`: NumPy array of shape `(bands, rows, columns)` representing Time 1.
* `image_b`: NumPy array of shape `(bands, rows, columns)` representing Time 2.
* `change_mask`: Boolean NumPy array of shape `(rows, columns)` where `True` denotes change (derived from OSCD pixel value 255).
* `dates`: Tuple of `(date_1, date_2)` parsed from `dates.txt`, or `None` if missing.
* `image_a_metadata` / `image_b_metadata`: [`RasterMetadata`](file:///d:/Earth%20ChangeAi/src/earthchange/cv/oscd_loader.py) containing:
  * `width`, `height`
  * `band_count`, `band_names`
  * `resolution`: pixel resolution `(x_res, y_res)` extracted from the georeferencing affine transform, or `None`
  * `crs`: Coordinate Reference System string, or `None`

---

## 5. Assumptions

1. **Pre-registration:** Image pairs in `imgs_1_rect` and `imgs_2_rect` are already co-registered by the dataset creators. The loader does not perform image registration.
2. **Binary Task:** Ground-truth masks contain binary values (0 = no change, 255 = change). Semantic question answering (e.g. distinguishing forest loss from construction) is not natively labeled in OSCD.
3. **No Synthetic Metadata:** If spatial resolution or CRS is missing from GeoTIFF headers, the loader sets `resolution = None` and `crs = None` rather than inventing metadata.

---

## 6. What Remains Unknown / Deferred

* **Full 13-Band Multi-Resolution Stacking:** Native 20m and 60m bands have lower pixel dimensions than 10m bands in raw OSCD. Stacking all 13 bands simultaneously requires spatial resampling (e.g., bicubic upsampling to 10m). The current loader requires bands within an image to have matching spatial dimensions.
* **Local Real Dataset Availability:** The full ~10GB OSCD archive is not currently downloaded in this local repository workspace. All verification is performed with synthetic fixtures matching the verified structure.
