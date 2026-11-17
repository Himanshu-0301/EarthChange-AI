"""Computer Vision module.

Responsible for visual measurement: change detection, masking, area calculation.
Must NOT interpret user questions or generate natural language.
"""

from earthchange.cv.cva import CVAResult, cva_otsu
from earthchange.cv.oscd_loader import (
    OSCDDatasetError,
    OSCDPair,
    RasterMetadata,
    load_oscd_pair,
)
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

__all__ = [
    "CVAResult",
    "ChangeDetectionConfig",
    "ChangeDetectionResult",
    "InputValidationError",
    "OSCDDatasetError",
    "OSCDPair",
    "REGION_SLICES",
    "RasterMetadata",
    "RegionStats",
    "analyze_all_regions",
    "analyze_region",
    "area_calculation",
    "cva_otsu",
    "detect_changes",
    "generate_visualization",
    "load_oscd_pair",
    "preprocess_images",
    "validate_images",
]
