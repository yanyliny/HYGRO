#!/usr/bin/env python3
"""
Droplet Condensation Analysis from ilastik Probability Masks

This script analyzes droplet condensation using ilastik pixel classification
probability outputs stored in HDF5 files. It handles the common "halo problem"
where ilastik captures droplet edges but not filled interiors.

Features:
- Auto-detection of HDF5 dataset and axis arrangement
- Morphological operations to close and fill halo-shaped droplets
- Per-frame and per-droplet statistics exported to CSV
- PNG overlays for visual quality control

Author: Generated for HYGRO project
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import morphology, measure, segmentation
from skimage.morphology import disk, binary_closing, binary_dilation, binary_erosion
from skimage.morphology import reconstruction, remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import imageio.v3 as iio


# =============================================================================
# HDF5 and Data Loading Functions
# =============================================================================

def find_probability_dataset(h5file: h5py.File) -> Tuple[str, np.ndarray]:
    """
    Find and return the probability dataset from an ilastik HDF5 file.

    Tries 'exported_data' first, then 'probabilities', then falls back
    to the first available dataset.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    dataset_name : str
        Name of the dataset found.
    data : np.ndarray
        The probability data array.
    """
    preferred_keys = ["exported_data", "probabilities"]

    # Get all dataset keys (recursively find datasets)
    all_keys = []
    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_keys.append(name)
    h5file.visititems(collect_datasets)

    if not all_keys:
        raise ValueError("No datasets found in HDF5 file!")

    # Try preferred keys first
    for key in preferred_keys:
        if key in h5file:
            print(f"[INFO] Using dataset: '{key}'")
            return key, np.array(h5file[key])

    # Fall back to first dataset
    fallback_key = all_keys[0]
    print(f"[WARNING] Neither 'exported_data' nor 'probabilities' found.")
    print(f"[WARNING] Using first available dataset: '{fallback_key}'")
    return fallback_key, np.array(h5file[fallback_key])


def normalize_ilastik_axes(prob_volume: np.ndarray) -> np.ndarray:
    """
    Take raw ilastik probability array and return array of shape (T, Y, X, C),
    where T can be 1 if no time dimension is present.

    Handles common ilastik output shapes:
    - (Y, X, C) -> single 2D frame
    - (T, Y, X, C) -> 2D time-lapse
    - (T, Z, Y, X, C) -> 3D time-lapse (Z must be 1 or will be max-projected)
    - (Z, Y, X, C) -> single 3D frame (Z must be 1 or will be max-projected)

    Parameters
    ----------
    prob_volume : np.ndarray
        Raw probability volume from ilastik.

    Returns
    -------
    normalized : np.ndarray
        Array of shape (T, Y, X, C).
    """
    shape = prob_volume.shape
    ndim = prob_volume.ndim

    print(f"[INFO] Input probability shape: {shape}, dtype: {prob_volume.dtype}")

    if ndim == 3:
        # Shape: (Y, X, C) - single 2D frame
        # Add time dimension
        normalized = prob_volume[np.newaxis, ...]
        print(f"[INFO] Detected single 2D frame (Y, X, C). Added T dimension.")

    elif ndim == 4:
        # Could be (T, Y, X, C) or (Z, Y, X, C)
        # Heuristic: if first dimension is small and second/third are large,
        # likely (Z, Y, X, C). Otherwise assume (T, Y, X, C).
        # Most common case is (T, Y, X, C)

        # Check if it looks like Z dimension (typically Z is small, Y/X are large)
        if shape[0] <= 10 and shape[1] > 50 and shape[2] > 50:
            # Might be (Z, Y, X, C) - max project over Z
            print(f"[INFO] Ambiguous 4D shape. Treating as (Z, Y, X, C).")
            if shape[0] == 1:
                # Z=1, just squeeze it
                normalized = prob_volume[np.newaxis, 0, ...]
                print(f"[INFO] Z=1, squeezed to (1, Y, X, C).")
            else:
                # Max project over Z
                print(f"[WARNING] Z > 1 ({shape[0]}). Max-projecting over Z dimension.")
                normalized = np.max(prob_volume, axis=0, keepdims=False)
                normalized = normalized[np.newaxis, ...]
        else:
            # Assume (T, Y, X, C)
            normalized = prob_volume
            print(f"[INFO] Detected time-lapse (T, Y, X, C).")

    elif ndim == 5:
        # Shape: (T, Z, Y, X, C)
        t_dim, z_dim = shape[0], shape[1]

        if z_dim == 1:
            # Squeeze Z dimension
            normalized = prob_volume[:, 0, :, :, :]
            print(f"[INFO] Detected (T, Z, Y, X, C) with Z=1. Squeezed to (T, Y, X, C).")
        else:
            # Max project over Z for each time point
            print(f"[WARNING] Z > 1 ({z_dim}). Max-projecting over Z for each frame.")
            normalized = np.max(prob_volume, axis=1)

    else:
        raise ValueError(
            f"Unexpected number of dimensions: {ndim}. "
            f"Expected 3 (Y,X,C), 4 (T,Y,X,C), or 5 (T,Z,Y,X,C). "
            f"Got shape: {shape}"
        )

    print(f"[INFO] Normalized shape: {normalized.shape} (T, Y, X, C)")
    return normalized


def load_raw_images(raw_path: str) -> Optional[np.ndarray]:
    """
    Load raw images from a file path (TIFF stack, PNG, etc.).

    Parameters
    ----------
    raw_path : str
        Path to the raw image file.

    Returns
    -------
    images : np.ndarray or None
        Array of shape (T, Y, X) or (T, Y, X, RGB). None if loading fails.
    """
    if not os.path.exists(raw_path):
        print(f"[WARNING] Raw image path does not exist: {raw_path}")
        return None

    try:
        # Use imageio to read (handles TIFF stacks, PNG sequences, etc.)
        images = iio.imread(raw_path)

        # Ensure we have a time dimension
        if images.ndim == 2:
            # Single grayscale image
            images = images[np.newaxis, ...]
        elif images.ndim == 3:
            # Could be (T, Y, X) or (Y, X, RGB)
            # If last dim is 3 or 4, likely RGB/RGBA
            if images.shape[-1] in [3, 4]:
                images = images[np.newaxis, ...]

        print(f"[INFO] Loaded raw images: shape={images.shape}, dtype={images.dtype}")
        return images

    except Exception as e:
        print(f"[WARNING] Failed to load raw images: {e}")
        return None


# =============================================================================
# Image Processing Functions
# =============================================================================

def extract_class_probability(prob_4d: np.ndarray, class_index: int) -> np.ndarray:
    """
    Extract probability map for a specific class.

    Parameters
    ----------
    prob_4d : np.ndarray
        Probability volume of shape (T, Y, X, C).
    class_index : int
        Index of the class to extract (0-based).

    Returns
    -------
    class_prob : np.ndarray
        Array of shape (T, Y, X) with probabilities for the specified class.
    """
    n_classes = prob_4d.shape[-1]
    if class_index < 0 or class_index >= n_classes:
        raise ValueError(
            f"class_index {class_index} out of range. "
            f"Available classes: 0 to {n_classes - 1}"
        )

    class_prob = prob_4d[..., class_index]
    print(f"[INFO] Extracted class {class_index} probability: "
          f"min={class_prob.min():.3f}, max={class_prob.max():.3f}")

    return class_prob


def find_droplet_centers_from_halos(
    prob_frame: np.ndarray,
    edge_threshold: float = 0.5,
    center_threshold: float = 0.3,
    min_center_distance: int = 10
) -> np.ndarray:
    """
    Find droplet centers by detecting the low-probability regions inside high-probability rings.

    Halo pattern: droplets have bright edges (high prob) and dark centers (low prob).
    We find centers by looking for local minima in probability that are surrounded by edges.

    Parameters
    ----------
    prob_frame : np.ndarray
        2D probability map (Y, X).
    edge_threshold : float
        Threshold above which pixels are considered droplet edges.
    center_threshold : float
        Threshold below which pixels could be droplet centers.
    min_center_distance : int
        Minimum distance between detected centers.

    Returns
    -------
    markers : np.ndarray
        Labeled marker image where each droplet center has a unique label.
    """
    # Get the edge mask (high probability ring regions)
    edges = prob_frame > edge_threshold

    # Get potential center regions (low probability inside droplets)
    # These are pixels with low probability
    potential_centers = prob_frame < center_threshold

    # Find regions that are enclosed by edges (holes in the edge mask)
    # Use binary_fill_holes to find what's inside closed rings
    filled_edges = binary_fill_holes(edges)

    # The interior of droplets = filled - edges
    interior = filled_edges & ~edges

    # Centers must be in low-probability regions AND inside filled edge regions
    center_candidates = potential_centers & interior

    # Clean up small noise
    center_candidates = remove_small_objects(center_candidates, min_size=5)

    # Use distance transform to find the center of each candidate region
    if np.any(center_candidates):
        distance = distance_transform_edt(center_candidates)

        # Find local maxima of distance transform as droplet centers
        coords = peak_local_max(
            distance,
            min_distance=min_center_distance,
            labels=label(center_candidates),
            exclude_border=False
        )

        # Create marker image
        markers = np.zeros(prob_frame.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
    else:
        markers = np.zeros(prob_frame.shape, dtype=np.int32)

    return markers


def fill_holes_per_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in each connected component individually.

    Unlike binary_fill_holes which fills globally, this fills holes
    within each connected component separately, preventing adjacent
    components from merging.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask with potential holes.

    Returns
    -------
    filled : np.ndarray
        Binary mask with holes filled per-component.
    """
    # Label connected components
    labeled = label(binary_mask, connectivity=2)
    n_components = labeled.max()

    if n_components == 0:
        return binary_mask.copy()

    filled = np.zeros_like(binary_mask)

    for comp_id in range(1, n_components + 1):
        # Extract this component
        component = (labeled == comp_id)

        # Get bounding box for efficiency
        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding for fill_holes to work correctly
        pad = 2
        rmin_p = max(0, rmin - pad)
        rmax_p = min(binary_mask.shape[0], rmax + pad + 1)
        cmin_p = max(0, cmin - pad)
        cmax_p = min(binary_mask.shape[1], cmax + pad + 1)

        # Extract region and fill holes
        region = component[rmin_p:rmax_p, cmin_p:cmax_p]
        region_filled = binary_fill_holes(region)

        # Put back
        filled[rmin_p:rmax_p, cmin_p:cmax_p] |= region_filled

    return filled.astype(np.uint8)


def segment_halo_droplets_watershed(
    prob_frame: np.ndarray,
    edge_threshold: float = 0.5,
    center_threshold: float = 0.3,
    min_center_distance: int = 10,
    closing_radius: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment halo-shaped droplets using watershed from detected centers.

    This handles the case where ilastik produces ring/halo patterns with
    bright edges and dark centers. Uses watershed to separate touching droplets.

    Strategy:
    1. Find droplet centers using distance transform on low-probability regions
    2. Use watershed from centers, constrained by the probability map
    3. This separates touching droplets even if their edges connect

    Parameters
    ----------
    prob_frame : np.ndarray
        2D probability map (Y, X).
    edge_threshold : float
        Threshold for detecting droplet edges.
    center_threshold : float
        Threshold for detecting droplet centers.
    min_center_distance : int
        Minimum distance between droplet centers.
    closing_radius : int
        Radius for closing gaps in edges.

    Returns
    -------
    binary_mask : np.ndarray
        Binary mask of all droplets.
    labeled_mask : np.ndarray
        Labeled mask where each droplet has unique integer.
    """
    # Step 1: Get edge regions (high probability = droplet edges in halo pattern)
    edges = prob_frame > edge_threshold

    # Step 2: Find droplet centers using distance transform from edges
    # In halo pattern, center is far from the bright edges
    # Use inverted probability: low prob regions are "high" in inverted
    inverted_prob = 1.0 - prob_frame

    # Distance from edge pixels - centers are far from edges
    distance_from_edges = distance_transform_edt(~edges)

    # Weight by inverted probability (prefer low-prob centers)
    center_score = distance_from_edges * (inverted_prob > (1 - center_threshold))

    # Find local maxima as droplet centers
    if np.any(center_score > 0):
        coords = peak_local_max(
            center_score,
            min_distance=min_center_distance,
            threshold_abs=1.0,  # Must be at least 1 pixel from edge
            exclude_border=False
        )

        # Create marker image
        markers = np.zeros(prob_frame.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        n_markers = len(coords)
    else:
        markers = np.zeros(prob_frame.shape, dtype=np.int32)
        n_markers = 0

    if n_markers == 0:
        # Fall back to simple edge-based segmentation
        if closing_radius > 0:
            edges = binary_closing(edges, footprint=disk(closing_radius))
        filled = fill_holes_per_component(edges)
        labeled = label(filled, connectivity=2)
        binary_mask = filled.astype(np.uint8)
        return binary_mask, labeled

    # Step 3: Create mask - use probability threshold directly
    # Any pixel with prob > low_threshold could be part of a droplet
    low_threshold = edge_threshold * 0.3  # Be generous
    droplet_mask = prob_frame > low_threshold

    # Step 4: Use watershed to separate touching droplets
    # Elevation = probability (high prob = ridges/barriers between droplets)
    # Watershed grows from centers (low prob) and stops at edges (high prob)
    elevation = prob_frame.copy()

    # Dilate markers slightly
    markers_dilated = binary_dilation(markers > 0, footprint=disk(3))
    markers_labeled = label(markers_dilated)

    # Re-assign original marker labels
    final_markers = np.zeros_like(markers)
    for i in range(1, n_markers + 1):
        orig_coords = np.where(markers == i)
        if len(orig_coords[0]) > 0:
            y, x = orig_coords[0][0], orig_coords[1][0]
            region_label = markers_labeled[y, x]
            if region_label > 0:
                final_markers[markers_labeled == region_label] = i

    # Watershed - will naturally stop at high probability boundaries
    labeled = watershed(elevation, final_markers, mask=droplet_mask)

    binary_mask = (labeled > 0).astype(np.uint8)

    return binary_mask, labeled


def segment_droplets_inverted(
    prob_frame: np.ndarray,
    prob_threshold: float = 0.5,
    min_droplet_size: int = 50,
    max_hole_size: int = 500,
    separation_threshold: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternative segmentation: treat LOW probability as droplet interiors.

    For halo patterns where the CENTER of droplets has low probability,
    we can invert the logic: threshold to find dark centers, then expand.

    Parameters
    ----------
    prob_frame : np.ndarray
        2D probability map (Y, X).
    prob_threshold : float
        Pixels BELOW this are considered droplet interiors.
    min_droplet_size : int
        Minimum droplet area.
    max_hole_size : int
        Maximum hole size to fill in droplets.
    separation_threshold : float
        Higher threshold to find definite edges for separation.

    Returns
    -------
    binary_mask : np.ndarray
        Binary mask of droplets.
    labeled_mask : np.ndarray
        Labeled droplets.
    """
    # Find definite droplet regions (low probability = inside droplet)
    droplet_interior = prob_frame < prob_threshold

    # Find high-confidence edges to use as barriers
    definite_edges = prob_frame > separation_threshold

    # Remove the edges from interior estimate
    droplet_interior = droplet_interior & ~definite_edges

    # Clean up
    droplet_interior = remove_small_objects(droplet_interior, min_size=min_droplet_size // 2)

    # Label connected components
    labeled = label(droplet_interior, connectivity=2)

    # For each component, dilate slightly to capture the edge region
    # but use watershed to prevent merging

    # Distance transform for watershed
    distance = distance_transform_edt(droplet_interior)

    # Use the labeled regions as markers
    markers = labeled.copy()

    # Define the basin - everywhere that's not a definite edge
    basin = ~definite_edges

    # Watershed to expand droplets up to edges
    expanded = watershed(-distance, markers, mask=basin)

    # Fill small holes in each droplet
    binary_mask = (expanded > 0).astype(np.uint8)

    return binary_mask, expanded


def process_single_frame(
    prob_frame: np.ndarray,
    prob_threshold: float,
    min_area: int,
    max_area: int,
    closing_radius: int,
    min_center_distance: int = 10,
    center_threshold: float = 0.3,
    use_watershed: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single probability frame to get labeled droplets.

    Uses watershed-based segmentation to properly separate halo-shaped droplets.

    Parameters
    ----------
    prob_frame : np.ndarray
        2D probability map (Y, X).
    prob_threshold : float
        Threshold for detecting droplet edges.
    min_area : int
        Minimum droplet area to keep.
    max_area : int
        Maximum droplet area (0 = no limit).
    closing_radius : int
        Radius for morphological closing.
    min_center_distance : int
        Minimum distance between droplet centers for watershed.
    center_threshold : float
        Probability threshold below which pixels are considered droplet centers.
    use_watershed : bool
        If True, use watershed segmentation. If False, use simple thresholding.

    Returns
    -------
    binary_mask : np.ndarray
        Final binary mask after processing.
    labeled_mask : np.ndarray
        Labeled regions (each droplet has unique integer label).
    """
    if use_watershed:
        # Use watershed-based segmentation for halo droplets
        binary, labeled = segment_halo_droplets_watershed(
            prob_frame,
            edge_threshold=prob_threshold,
            center_threshold=center_threshold,
            min_center_distance=min_center_distance,
            closing_radius=closing_radius
        )
    else:
        # Simple thresholding fallback
        binary = prob_frame > prob_threshold
        binary = binary_closing(binary, footprint=disk(closing_radius))
        binary = binary_fill_holes(binary)
        labeled = label(binary, connectivity=2)
        binary = binary.astype(np.uint8)

    # Filter by area
    regions = regionprops(labeled)
    for region in regions:
        area = region.area
        # Remove if too small
        if area < min_area:
            labeled[labeled == region.label] = 0
        # Remove if too large (and max_area is specified)
        elif max_area > 0 and area > max_area:
            labeled[labeled == region.label] = 0

    # Relabel to ensure contiguous labels
    labeled = label(labeled > 0, connectivity=2)
    binary = (labeled > 0).astype(np.uint8)

    return binary, labeled


# =============================================================================
# Statistics and Metrics
# =============================================================================

def compute_frame_statistics(
    labeled_mask: np.ndarray,
    frame_index: int,
    image_area: int
) -> Dict[str, Any]:
    """
    Compute per-frame statistics.

    Parameters
    ----------
    labeled_mask : np.ndarray
        Labeled droplet mask.
    frame_index : int
        Index of the frame.
    image_area : int
        Total image area in pixels.

    Returns
    -------
    stats : dict
        Dictionary of frame-level statistics.
    """
    regions = regionprops(labeled_mask)

    n_droplets = len(regions)
    total_droplet_area = sum(r.area for r in regions)
    coverage_fraction = total_droplet_area / image_area if image_area > 0 else 0

    if n_droplets > 0:
        areas = [r.area for r in regions]
        mean_area = np.mean(areas)
        median_area = np.median(areas)
        std_area = np.std(areas)
        min_droplet_area = min(areas)
        max_droplet_area = max(areas)
    else:
        mean_area = median_area = std_area = 0
        min_droplet_area = max_droplet_area = 0

    return {
        'frame': frame_index,
        'n_droplets': n_droplets,
        'total_droplet_area_px': total_droplet_area,
        'coverage_fraction': coverage_fraction,
        'coverage_percent': coverage_fraction * 100,
        'mean_area_px': mean_area,
        'median_area_px': median_area,
        'std_area_px': std_area,
        'min_droplet_area_px': min_droplet_area,
        'max_droplet_area_px': max_droplet_area,
    }


def compute_droplet_statistics(
    labeled_mask: np.ndarray,
    frame_index: int
) -> List[Dict[str, Any]]:
    """
    Compute per-droplet statistics for a single frame.

    Parameters
    ----------
    labeled_mask : np.ndarray
        Labeled droplet mask.
    frame_index : int
        Index of the frame.

    Returns
    -------
    droplet_stats : list of dict
        List of dictionaries with per-droplet metrics.
    """
    # Use regionprops_table for efficiency
    props = regionprops_table(
        labeled_mask,
        properties=[
            'label', 'area', 'centroid', 'eccentricity',
            'equivalent_diameter_area', 'perimeter', 'solidity',
            'major_axis_length', 'minor_axis_length'
        ]
    )

    droplet_list = []
    n_droplets = len(props['label'])

    for i in range(n_droplets):
        droplet_list.append({
            'frame': frame_index,
            'droplet_id': int(props['label'][i]),
            'area_px': props['area'][i],
            'centroid_y': props['centroid-0'][i],
            'centroid_x': props['centroid-1'][i],
            'eccentricity': props['eccentricity'][i],
            'equivalent_diameter_px': props['equivalent_diameter_area'][i],
            'perimeter_px': props['perimeter'][i],
            'solidity': props['solidity'][i],
            'major_axis_px': props['major_axis_length'][i],
            'minor_axis_px': props['minor_axis_length'][i],
        })

    return droplet_list


# =============================================================================
# Visualization Functions
# =============================================================================

def create_overlay_image(
    raw_frame: Optional[np.ndarray],
    prob_frame: np.ndarray,
    binary_mask: np.ndarray,
    labeled_mask: np.ndarray,
    frame_index: int,
    frame_stats: Dict[str, Any]
) -> plt.Figure:
    """
    Create a figure with overlay visualization for QC.

    Parameters
    ----------
    raw_frame : np.ndarray or None
        Raw image frame (Y, X) or (Y, X, 3). None if not available.
    prob_frame : np.ndarray
        Probability map (Y, X).
    binary_mask : np.ndarray
        Binary droplet mask.
    labeled_mask : np.ndarray
        Labeled droplet mask.
    frame_index : int
        Frame number.
    frame_stats : dict
        Frame-level statistics for annotation.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure with the overlay.
    """
    n_cols = 3 if raw_frame is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    col_idx = 0

    # Raw image with overlay (if available)
    if raw_frame is not None:
        ax = axes[col_idx]
        col_idx += 1

        # Normalize raw frame for display
        if raw_frame.ndim == 2:
            display_raw = raw_frame
            ax.imshow(display_raw, cmap='gray')
        else:
            display_raw = raw_frame
            ax.imshow(display_raw)

        # Overlay mask contours
        ax.contour(binary_mask, colors='cyan', linewidths=0.8, levels=[0.5])
        ax.set_title(f'Frame {frame_index}: Raw + Contours')
        ax.axis('off')

    # Probability map
    ax = axes[col_idx]
    col_idx += 1
    im = ax.imshow(prob_frame, cmap='viridis', vmin=0, vmax=1)
    ax.contour(binary_mask, colors='red', linewidths=0.5, levels=[0.5])
    ax.set_title(f'Probability Map')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labeled mask with colormap
    ax = axes[col_idx]
    n_labels = labeled_mask.max()
    if n_labels > 0:
        # Create random colormap for labels
        np.random.seed(42)  # Reproducible colors
        colors = np.random.rand(n_labels + 1, 3)
        colors[0] = [0, 0, 0]  # Background is black
        cmap = ListedColormap(colors)
        ax.imshow(labeled_mask, cmap=cmap, interpolation='nearest')
    else:
        ax.imshow(labeled_mask, cmap='gray')

    ax.set_title(f'Labeled Droplets (n={frame_stats["n_droplets"]})')
    ax.axis('off')

    # Add statistics annotation
    stats_text = (
        f"Droplets: {frame_stats['n_droplets']}\n"
        f"Coverage: {frame_stats['coverage_percent']:.2f}%\n"
        f"Mean area: {frame_stats['mean_area_px']:.1f} px"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_mask_only_image(
    binary_mask: np.ndarray,
    labeled_mask: np.ndarray,
    frame_index: int,
    frame_stats: Dict[str, Any]
) -> plt.Figure:
    """
    Create a figure with just the mask visualization (no overlay).

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary droplet mask.
    labeled_mask : np.ndarray
        Labeled droplet mask.
    frame_index : int
        Frame number.
    frame_stats : dict
        Frame-level statistics.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Binary mask
    axes[0].imshow(binary_mask, cmap='gray')
    axes[0].set_title(f'Frame {frame_index}: Binary Mask')
    axes[0].axis('off')

    # Labeled mask
    n_labels = labeled_mask.max()
    if n_labels > 0:
        np.random.seed(42)
        colors = np.random.rand(n_labels + 1, 3)
        colors[0] = [0, 0, 0]
        cmap = ListedColormap(colors)
        axes[1].imshow(labeled_mask, cmap=cmap, interpolation='nearest')
    else:
        axes[1].imshow(labeled_mask, cmap='gray')

    axes[1].set_title(f'Labeled Droplets (n={frame_stats["n_droplets"]})')
    axes[1].axis('off')

    # Statistics annotation
    stats_text = (
        f"Droplets: {frame_stats['n_droplets']}\n"
        f"Coverage: {frame_stats['coverage_percent']:.2f}%\n"
        f"Mean area: {frame_stats['mean_area_px']:.1f} px"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def run_analysis(args: argparse.Namespace) -> None:
    """
    Main analysis pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / 'pngs'
    png_dir.mkdir(exist_ok=True)

    use_watershed = not args.no_watershed

    print("=" * 60)
    print("DROPLET CONDENSATION ANALYSIS")
    print("=" * 60)
    print(f"H5 file: {args.h5}")
    print(f"Output directory: {output_dir}")
    print(f"Class index: {args.class_index}")
    print(f"Segmentation: {'watershed' if use_watershed else 'simple threshold'}")
    print(f"Edge threshold: {args.prob_threshold}")
    print(f"Center threshold: {args.center_threshold}")
    print(f"Min center distance: {args.min_center_distance} px")
    print(f"Min area: {args.min_area} px")
    print(f"Max area: {args.max_area} px" if args.max_area > 0 else "Max area: unlimited")
    print(f"Closing radius: {args.closing_radius}")
    print("=" * 60)

    # Load HDF5 probability data
    print("\n[STEP 1] Loading probability data...")
    with h5py.File(args.h5, 'r') as h5file:
        dataset_name, raw_data = find_probability_dataset(h5file)

    # Normalize axes to (T, Y, X, C)
    prob_4d = normalize_ilastik_axes(raw_data)

    # Extract the specified class
    class_prob = extract_class_probability(prob_4d, args.class_index)
    n_frames, height, width = class_prob.shape
    image_area = height * width

    print(f"\n[INFO] Processing {n_frames} frame(s) of size {height} x {width}")

    # Load raw images if provided
    raw_images = None
    if args.raw:
        print(f"\n[STEP 2] Loading raw images from: {args.raw}")
        raw_images = load_raw_images(args.raw)

        if raw_images is not None:
            # Check frame count matches
            if raw_images.shape[0] != n_frames:
                print(f"[WARNING] Raw image count ({raw_images.shape[0]}) != "
                      f"probability frames ({n_frames}). Using min of both.")
                n_frames = min(n_frames, raw_images.shape[0])

    # Process each frame
    print(f"\n[STEP 3] Processing frames...")

    frame_stats_list = []
    droplet_stats_list = []

    for frame_idx in range(n_frames):
        if frame_idx % 10 == 0 or frame_idx == n_frames - 1:
            print(f"  Processing frame {frame_idx + 1}/{n_frames}...")

        prob_frame = class_prob[frame_idx]

        # Process frame
        binary_mask, labeled_mask = process_single_frame(
            prob_frame,
            prob_threshold=args.prob_threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            closing_radius=args.closing_radius,
            min_center_distance=args.min_center_distance,
            center_threshold=args.center_threshold,
            use_watershed=use_watershed
        )

        # Compute statistics
        frame_stats = compute_frame_statistics(labeled_mask, frame_idx, image_area)
        frame_stats_list.append(frame_stats)

        droplet_stats = compute_droplet_statistics(labeled_mask, frame_idx)
        droplet_stats_list.extend(droplet_stats)

        # Save PNG if needed
        if frame_idx % args.png_stride == 0:
            raw_frame = raw_images[frame_idx] if raw_images is not None else None

            if args.no_overlays:
                fig = create_mask_only_image(
                    binary_mask, labeled_mask, frame_idx, frame_stats
                )
            else:
                fig = create_overlay_image(
                    raw_frame, prob_frame, binary_mask, labeled_mask,
                    frame_idx, frame_stats
                )

            png_path = png_dir / f'frame_{frame_idx:04d}.png'
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    # Save CSV files
    print(f"\n[STEP 4] Saving results...")

    # Frame-level statistics
    frame_df = pd.DataFrame(frame_stats_list)
    frame_csv_path = output_dir / 'frame_statistics.csv'
    frame_df.to_csv(frame_csv_path, index=False)
    print(f"  Saved frame statistics: {frame_csv_path}")

    # Droplet-level statistics
    if droplet_stats_list:
        droplet_df = pd.DataFrame(droplet_stats_list)
        droplet_csv_path = output_dir / 'droplet_statistics.csv'
        droplet_df.to_csv(droplet_csv_path, index=False)
        print(f"  Saved droplet statistics: {droplet_csv_path}")
    else:
        print("  [WARNING] No droplets detected in any frame. Skipping droplet CSV.")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {n_frames}")
    print(f"Total droplets detected: {len(droplet_stats_list)}")

    if frame_stats_list:
        avg_coverage = np.mean([s['coverage_percent'] for s in frame_stats_list])
        avg_droplets = np.mean([s['n_droplets'] for s in frame_stats_list])
        print(f"Average coverage: {avg_coverage:.2f}%")
        print(f"Average droplets per frame: {avg_droplets:.1f}")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - frame_statistics.csv")
    if droplet_stats_list:
        print(f"  - droplet_statistics.csv")
    print(f"  - pngs/ ({len(list(png_dir.glob('*.png')))} images)")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze droplet condensation from ilastik probability masks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required arguments (uses watershed segmentation by default)
  python droplet_ilastik_analysis.py --h5 probs.h5 --class-index 0

  # Tune for halo-shaped droplets (adjust center detection)
  python droplet_ilastik_analysis.py --h5 probs.h5 --class-index 0 \\
      --prob-threshold 0.6 --center-threshold 0.3 --min-center-distance 15

  # With raw images for overlay visualization
  python droplet_ilastik_analysis.py --h5 probs.h5 --class-index 0 --raw images.tif

  # Filter droplet sizes
  python droplet_ilastik_analysis.py --h5 probs.h5 --class-index 0 \\
      --min-area 50 --max-area 10000

  # Disable watershed (simple thresholding)
  python droplet_ilastik_analysis.py --h5 probs.h5 --class-index 0 --no-watershed

  # Save every 5th frame only
  python droplet_ilastik_analysis.py --h5 probs.h5 --class-index 0 --png-stride 5
        """
    )

    # Required arguments
    parser.add_argument(
        '--h5',
        type=str,
        required=True,
        help='Path to ilastik HDF5 probability output file.'
    )
    parser.add_argument(
        '--class-index',
        type=int,
        required=True,
        help='Index of the droplet class channel (0-based).'
    )

    # Optional arguments
    parser.add_argument(
        '--raw',
        type=str,
        default=None,
        help='Path to raw image stack (tif, png, etc.) for overlay visualization.'
    )
    parser.add_argument(
        '--prob-threshold',
        type=float,
        default=0.5,
        help='Probability threshold for detecting droplet edges (default: 0.5).'
    )
    parser.add_argument(
        '--center-threshold',
        type=float,
        default=0.3,
        help='Probability threshold below which pixels are droplet centers (default: 0.3). '
             'For halo patterns, centers have LOW probability.'
    )
    parser.add_argument(
        '--min-center-distance',
        type=int,
        default=10,
        help='Minimum distance between droplet centers in pixels (default: 10). '
             'Increase for larger/sparser droplets.'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=20,
        help='Minimum droplet area in pixels (default: 20).'
    )
    parser.add_argument(
        '--max-area',
        type=int,
        default=0,
        help='Maximum droplet area in pixels. 0 = no limit (default: 0).'
    )
    parser.add_argument(
        '--closing-radius',
        type=int,
        default=2,
        help='Disk radius for morphological closing (default: 2). '
             'Smaller values prevent merging adjacent droplets.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./droplet_output',
        help='Output directory for CSV and PNGs (default: ./droplet_output).'
    )
    parser.add_argument(
        '--png-stride',
        type=int,
        default=1,
        help='Save PNG for every Nth frame (default: 1 = every frame).'
    )
    parser.add_argument(
        '--no-overlays',
        action='store_true',
        help='Skip overlay images, save masks only.'
    )
    parser.add_argument(
        '--no-watershed',
        action='store_true',
        help='Disable watershed segmentation, use simple thresholding instead.'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate inputs
    if not os.path.exists(args.h5):
        print(f"[ERROR] HDF5 file not found: {args.h5}")
        sys.exit(1)

    if args.prob_threshold < 0 or args.prob_threshold > 1:
        print(f"[ERROR] prob_threshold must be between 0 and 1, got {args.prob_threshold}")
        sys.exit(1)

    if args.min_area < 0:
        print(f"[ERROR] min_area must be >= 0, got {args.min_area}")
        sys.exit(1)

    if args.max_area < 0:
        print(f"[ERROR] max_area must be >= 0, got {args.max_area}")
        sys.exit(1)

    if args.closing_radius < 0:
        print(f"[ERROR] closing_radius must be >= 0, got {args.closing_radius}")
        sys.exit(1)

    if args.center_threshold < 0 or args.center_threshold > 1:
        print(f"[ERROR] center_threshold must be between 0 and 1, got {args.center_threshold}")
        sys.exit(1)

    if args.min_center_distance < 1:
        print(f"[ERROR] min_center_distance must be >= 1, got {args.min_center_distance}")
        sys.exit(1)

    if args.png_stride < 1:
        print(f"[ERROR] png_stride must be >= 1, got {args.png_stride}")
        sys.exit(1)

    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

    # Run analysis
    try:
        run_analysis(args)
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
