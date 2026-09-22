"""
Data Utility Functions for River Morphology Prediction
======================================================

This module provides utility functions for:
- Loading and processing satellite imagery
- Handling GeoTIFF files
- Data normalization and preprocessing
- Temporal data management
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import yaml


def validate_path(value: str, name: str) -> Path:
    """Validate and return a non-empty filesystem path supplied by a caller."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"{name} must be a non-empty path string")
    return Path(value)


def validate_safe_pattern(pattern: str) -> str:
    """Validate a local filename pattern and reject traversal or absolute paths."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError("pattern must be a relative local filename pattern")
    return pattern


def load_config(config_path: str = "../config/config.yaml") -> Dict:
    """
    Load configuration from YAML file.

    Parameters:
    -----------
    config_path : str
        Path to configuration file

    Returns:
    --------
    dict : Configuration dictionary
    """
    path = validate_path(config_path, "config_path")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("config_path must point to a YAML file")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("configuration file must contain a top-level mapping")
    return config


def load_geotiff(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a GeoTIFF file and return the array and metadata.

    Parameters:
    -----------
    file_path : str
        Path to GeoTIFF file

    Returns:
    --------
    tuple : (array, metadata_dict)
        - array: numpy array of shape (H, W)
        - metadata: dictionary containing spatial reference info
    """
    path = validate_path(file_path, "file_path")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("file_path must point to a GeoTIFF file")
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": src.bounds,
            "dtype": src.dtypes[0],
        }
    return array, metadata


def save_geotiff(
    array: np.ndarray,
    output_path: str,
    metadata: Dict,
    dtype: str = "uint8",
    nodata_value: Optional[float] = None,
) -> None:
    """
    Save a numpy array as GeoTIFF with spatial reference.

    Parameters:
    -----------
    array : np.ndarray
        Array to save (H, W)
    output_path : str
        Output file path
    metadata : dict
        Metadata dictionary from load_geotiff
    dtype : str
        Output data type
    nodata_value : float, optional
        No data value
    """
    output = validate_path(output_path, "output_path")
    if output.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("output_path must point to a GeoTIFF file")
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("array must be a two-dimensional NumPy array")
    if not isinstance(metadata, dict) or "crs" not in metadata or "transform" not in metadata:
        raise ValueError("metadata must contain 'crs' and 'transform'")
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a NumPy dtype string")

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Set nodata value if not provided
    if nodata_value is None:
        if np.issubdtype(np.dtype(dtype), np.integer):
            nodata_value = 255 if "uint" in dtype else -9999
        else:
            nodata_value = -9999.0

    # Create output file
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=metadata["crs"],
        transform=metadata["transform"],
        nodata=nodata_value,
        compress="lzw",
    ) as dst:
        # Replace NaN with nodata value
        output_array = array.copy()
        output_array[np.isnan(output_array)] = nodata_value
        dst.write(output_array.astype(dtype), 1)


def load_temporal_sequence(
    directory: str, years: Optional[List[int]] = None, pattern: str = "*.tif"
) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Load a temporal sequence of GeoTIFF files.

    Parameters:
    -----------
    directory : str
        Directory containing GeoTIFF files
    years : list of int, optional
        Specific years to load. If None, loads all.
    pattern : str
        File pattern to match

    Returns:
    --------
    tuple : (sequence_array, years_list, metadata)
        - sequence_array: numpy array of shape (T, H, W)
        - years_list: list of years corresponding to each time step
        - metadata: metadata from first file
    """
    dir_path = validate_path(directory, "directory")
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {directory!r}")
    pattern = validate_safe_pattern(pattern)
    if years is not None and (
        not isinstance(years, (list, tuple, set))
        or not all(isinstance(year, int) and not isinstance(year, bool) for year in years)
    ):
        raise TypeError("years must be a sequence of integers")
    files = sorted(dir_path.glob(pattern))

    if not files:
        raise ValueError(f"No files found in {directory} matching {pattern}")

    # Load first file to get shape and metadata
    first_array, metadata = load_geotiff(str(files[0]))
    h, w = first_array.shape

    # Filter by years if specified
    if years is not None:
        files = [f for f in files if any(str(y) in f.name for y in years)]

    # Initialize sequence array
    sequence = np.zeros((len(files), h, w), dtype=np.float32)
    file_years = []

    # Load all files
    for i, file_path in enumerate(files):
        array, _ = load_geotiff(str(file_path))
        sequence[i] = array

        # Extract year from filename
        year = extract_year_from_filename(file_path.name)
        file_years.append(year)

    return sequence, file_years, metadata


def extract_year_from_filename(filename: str) -> int:
    """
    Extract year from filename.

    Parameters:
    -----------
    filename : str
        Filename containing year

    Returns:
    --------
    int : Extracted year
    """
    import re

    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"Could not extract year from filename: {filename}")


def normalize_array(
    array: np.ndarray, method: str = "minmax", feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normalize array values.

    Parameters:
    -----------
    array : np.ndarray
        Input array
    method : str
        Normalization method ('minmax', 'zscore', or 'none')
    feature_range : tuple
        Range for minmax normalization

    Returns:
    --------
    np.ndarray : Normalized array
    """
    if method == "none":
        return array

    elif method == "minmax":
        min_val, max_val = feature_range
        array_min = np.nanmin(array)
        array_max = np.nanmax(array)

        if array_max - array_min == 0:
            return np.full_like(array, min_val)

        normalized = (array - array_min) / (array_max - array_min)
        normalized = normalized * (max_val - min_val) + min_val
        return normalized

    elif method == "zscore":
        mean = np.nanmean(array)
        std = np.nanstd(array)

        if std == 0:
            return array - mean

        return (array - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_water_mask(
    array: np.ndarray, threshold: float = 0.0, method: str = "mndwi"
) -> np.ndarray:
    """
    Create binary water mask from continuous values.

    Parameters:
    -----------
    array : np.ndarray
        Input array (e.g., MNDWI values)
    threshold : float
        Threshold value
    method : str
        Method used ('mndwi', 'sar', etc.)

    Returns:
    --------
    np.ndarray : Binary mask (0=land, 1=water)
    """
    if method == "mndwi":
        # Water where MNDWI > threshold
        mask = (array > threshold).astype(np.uint8)
    elif method == "sar":
        # Water where VV < threshold (low backscatter)
        mask = (array < threshold).astype(np.uint8)
    else:
        raise ValueError(f"Unknown water detection method: {method}")

    return mask


def calculate_water_area(mask: np.ndarray, pixel_size_m: float = 60.0) -> float:
    """
    Calculate water area from binary mask.

    Parameters:
    -----------
    mask : np.ndarray
        Binary water mask
    pixel_size_m : float
        Pixel size in meters

    Returns:
    --------
    float : Water area in square kilometers
    """
    pixel_area_m2 = pixel_size_m**2
    water_pixels = np.sum(mask == 1)
    area_km2 = (water_pixels * pixel_area_m2) / 1e6
    return area_km2


def split_temporal_data(
    sequence: np.ndarray,
    years: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict:
    """
    Split temporal sequence into train/val/test sets.

    Parameters:
    -----------
    sequence : np.ndarray
        Temporal sequence (T, H, W)
    years : list
        Years corresponding to each time step
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio

    Returns:
    --------
    dict : Dictionary with 'train', 'val', 'test' keys containing
           {'data': array, 'years': list} for each split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n_samples = len(sequence)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    return {
        "train": {"data": sequence[:train_end], "years": years[:train_end]},
        "val": {"data": sequence[train_end:val_end], "years": years[train_end:val_end]},
        "test": {"data": sequence[val_end:], "years": years[val_end:]},
    }


def get_file_list(directory: str, extension: str = ".tif", sort: bool = True) -> List[Path]:
    """
    Get list of files in directory.

    Parameters:
    -----------
    directory : str
        Directory path
    extension : str
        File extension to filter
    sort : bool
        Whether to sort files

    Returns:
    --------
    list : List of Path objects
    """
    dir_path = Path(directory)
    pattern = f"*{extension}"
    files = list(dir_path.glob(pattern))

    if sort:
        files = sorted(files)

    return files


def compute_statistics(array: np.ndarray) -> Dict:
    """
    Compute basic statistics for array.

    Parameters:
    -----------
    array : np.ndarray
        Input array

    Returns:
    --------
    dict : Dictionary with statistics
    """
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "median": float(np.nanmedian(array)),
        "count": int(np.sum(~np.isnan(array))),
        "nan_count": int(np.sum(np.isnan(array))),
    }


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config()
    print(f"Loaded configuration for: {config['project']['name']}")

    # Example: Load a single GeoTIFF
    # array, metadata = load_geotiff("path/to/file.tif")
    # print(f"Loaded array shape: {array.shape}")
    # print(f"CRS: {metadata['crs']}")

    # Example: Load temporal sequence
    # sequence, years, metadata = load_temporal_sequence("data/yearly")
    # print(f"Loaded sequence shape: {sequence.shape}")
    # print(f"Years: {years}")
