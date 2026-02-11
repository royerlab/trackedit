"""InstanSeg inference integration for TrackEdit.

This module provides interactive cell segmentation using InstanSeg models.
It manages model loading, embedding caching, and inference execution for
click-based cell addition.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from napari.utils.notifications import show_warning

# Target pixel size for isotropic rescaling (must match training)
TARGET_PIXEL_SIZE = 0.5


def normalize(image: np.ndarray) -> np.ndarray:
    """Percentile-normalize a volume to [0, 1]."""
    data = image.astype(np.float32)
    low = np.percentile(data, 0.1)
    high = np.percentile(data, 99.9)
    if high > low:
        normalized = (data - low) / (high - low)
        # Clamp to [0, 1] to handle outliers above 99.9th percentile
        return np.clip(normalized, 0, 1)
    return data / (data.max() + 1e-8)


def rescale_to_isotropic(
    image: np.ndarray,
    current_scale: tuple,
    target_pixel_size: float = TARGET_PIXEL_SIZE,
    device: str = "cpu",
) -> Tuple[np.ndarray, tuple]:
    """Rescale a (Z, Y, X) volume to isotropic resolution using trilinear interpolation.

    Args:
        image: Input volume (Z, Y, X)
        current_scale: Current voxel sizes (z_size, y_size, x_size) in microns
        target_pixel_size: Target isotropic pixel size in microns
        device: Device to run interpolation on ('cuda' or 'cpu')

    Returns:
        (rescaled_image_np, scale_factors) where scale_factors = (sz, sy, sx)
    """
    sz = current_scale[0] / target_pixel_size
    sy = current_scale[1] / target_pixel_size
    sx = current_scale[2] / target_pixel_size

    d, h, w = image.shape
    new_d = int(round(d * sz))
    new_h = int(round(h * sy))
    new_w = int(round(w * sx))

    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
    if device == "cuda" and torch.cuda.is_available():
        tensor = tensor.to(device)
    rescaled = F.interpolate(
        tensor, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False
    )
    if device == "cuda":
        rescaled = rescaled.cpu()
    return rescaled.squeeze().numpy(), (sz, sy, sx)


def rescale_labels_back(labels: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Rescale labels from isotropic back to original shape using nearest interpolation."""
    tensor = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    rescaled = F.interpolate(tensor, size=original_shape, mode="nearest")
    return rescaled.squeeze().numpy().astype(np.int32)


def run_backbone(model, image_normalized: np.ndarray, device: str) -> torch.Tensor:
    """Run only the backbone. Returns (C, D, H, W) prediction tensor on GPU."""
    tensor = torch.from_numpy(image_normalized).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
    tensor = tensor.to(device)
    with torch.no_grad():
        pred = model.backbone(tensor)  # (1, C, D, H, W)
    return pred[0]  # (C, D, H, W), stays on device


def run_postprocessing(
    model,
    cached_pred: torch.Tensor,
    precomputed_seeds: Optional[torch.Tensor] = None,
    seed_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_threshold: float = 1.0,
    pairwise_filter_threshold: float = 0.0,
    peak_distance: int = 4,
    window_size: int = 64,
    use_iom: bool = False,
    use_pairwise_seed_filter: bool = False,
    cleanup_fragments: bool = True,
) -> np.ndarray:
    """Run only the postprocessing on a cached prediction.

    Args:
        model: InstanSeg TorchScript model
        cached_pred: Cached backbone embeddings (C, D, H, W) on GPU
        precomputed_seeds: Optional seed points in isotropic coordinates (N, 3)
        seed_threshold: Threshold for seed detection
        mask_threshold: Threshold for mask generation
        overlap_threshold: Threshold for overlap detection
        pairwise_filter_threshold: Threshold for pairwise filtering
        peak_distance: Distance for peak detection
        window_size: Size of processing window
        use_iom: Use intersection over minimum instead of IOU
        use_pairwise_seed_filter: Enable pairwise seed filtering
        cleanup_fragments: Remove small fragments

    Returns:
        Instance labels (Z, Y, X) as int32 numpy array
    """
    # Prepare seeds if provided
    seeds = None
    if precomputed_seeds is not None and precomputed_seeds.shape[0] > 0:
        seeds = precomputed_seeds.to(cached_pred.device)

    with torch.no_grad():
        labels = model.postprocessing(
            cached_pred,
            seed_threshold=seed_threshold,
            mask_threshold=mask_threshold,
            overlap_threshold=overlap_threshold,
            pairwise_filter_threshold=pairwise_filter_threshold,
            peak_distance=peak_distance,
            window_size=window_size,
            use_iom=use_iom,
            use_pairwise_seed_filter=use_pairwise_seed_filter,
            cleanup_fragments=cleanup_fragments,
            precomputed_seeds=seeds if seeds is not None else torch.empty(0),
        )  # (1, D, H, W)

    return labels.squeeze().cpu().numpy().astype(np.int32)


def extract_mask_at_seed(
    labels: np.ndarray, seed_position: tuple
) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """Extract the instance mask at seed position and compute its bounding box.

    Args:
        labels: Instance segmentation (Z, Y, X) with unique IDs per cell
        seed_position: (z, y, x) position in label coordinates

    Returns:
        (mask, bbox) where:
        - mask: Binary mask (Z, Y, X) of the cell at seed, or None if no cell
        - bbox: Bounding box ((z_min, y_min, x_min), (z_max, y_max, x_max)), or None
    """
    z, y, x = seed_position
    z, y, x = int(round(z)), int(round(y)), int(round(x))

    # Check bounds
    if not (
        0 <= z < labels.shape[0]
        and 0 <= y < labels.shape[1]
        and 0 <= x < labels.shape[2]
    ):
        return None, None

    # Get label ID at seed position
    label_id = labels[z, y, x]

    if label_id == 0:
        # No cell at this position
        return None, None

    # Extract binary mask for this instance
    mask = (labels == label_id).astype(bool)

    # Return the full mask (will be cropped after rescaling to original resolution)
    # Don't crop here because we need to rescale the full mask first
    return mask, None


class InstanSegInference:
    """Manages InstanSeg model and inference for TrackEdit.

    This class provides interactive cell segmentation by:
    1. Loading a TorchScript InstanSeg model
    2. Caching embeddings per time frame for fast repeated inference
    3. Running seed-based postprocessing on cached embeddings
    4. Extracting single-cell masks from instance segmentation
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        cache_size: int = 3,
    ):
        """Initialize InstanSeg inference engine.

        Args:
            model_path: Path to TorchScript (.pt) model file
            device: Device for inference ('cuda', 'cpu', or None for auto)
            cache_size: Maximum number of frames to cache embeddings for
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.cache_size = cache_size

        # Load model
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()

        # Cache structure: {time: (embeddings_tensor, iso_shape, scale_factors, original_shape)}
        self.cached_embeddings = {}
        self.cache_order = []  # Track insertion order for LRU eviction

        # Verify GPU setup
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✓ InstanSeg using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                print(
                    "⚠ Warning: CUDA requested but not available, falling back to CPU"
                )
                self.device = "cpu"
        else:
            print(f"InstanSeg using device: {device}")

    def run_inference_at_position(
        self,
        image_volume: np.ndarray,
        time_frame: int,
        position: tuple,
        scale: tuple,
    ) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        """Run InstanSeg inference with seed at position.

        Args:
            image_volume: Raw image volume (Z, Y, X) at original resolution
            time_frame: Time frame index (for caching)
            position: (z, y, x) seed position in original data coordinates
            scale: (z_scale, y_scale, x_scale) voxel sizes in microns

        Returns:
            (mask, bbox) tuple:
            - mask: Binary mask (Z, Y, X) at original resolution, or None if no cell
            - bbox: Bounding box ((z_min, y_min, x_min), (z_max, y_max, x_max)), or None
        """
        try:
            # Normalize image
            image_normalized = normalize(image_volume)

            # Rescale to isotropic resolution (on GPU if available)
            image_isotropic, scale_factors = rescale_to_isotropic(
                image_normalized, scale, TARGET_PIXEL_SIZE, device=self.device
            )
            iso_shape = image_isotropic.shape
            original_shape = image_volume.shape

            # Check cache for embeddings
            if time_frame in self.cached_embeddings:
                embeddings, cached_iso_shape, _, _ = self.cached_embeddings[time_frame]

                # Verify shape matches (in case scale changed)
                if cached_iso_shape != iso_shape:
                    embeddings = self._compute_and_cache_embeddings(
                        time_frame,
                        image_isotropic,
                        iso_shape,
                        scale_factors,
                        original_shape,
                    )
            else:
                # Run backbone and cache
                embeddings = self._compute_and_cache_embeddings(
                    time_frame,
                    image_isotropic,
                    iso_shape,
                    scale_factors,
                    original_shape,
                )

            # Convert position to isotropic coordinates
            position_iso = self._convert_to_isotropic_coords(
                position, original_shape, iso_shape
            )

            # Run postprocessing with seed
            seed_tensor = torch.from_numpy(np.array([position_iso])).float()  # (1, 3)

            labels_isotropic = run_postprocessing(
                self.model,
                embeddings,
                precomputed_seeds=seed_tensor,
                seed_threshold=0.5,
                mask_threshold=0.5,
                overlap_threshold=1.0,
                pairwise_filter_threshold=0.0,
                peak_distance=4,
                window_size=64,
                use_iom=False,
                use_pairwise_seed_filter=False,
                cleanup_fragments=True,
            )

            # Extract single-instance mask at seed
            mask_isotropic, _ = extract_mask_at_seed(labels_isotropic, position_iso)

            if mask_isotropic is None:
                show_warning("No cell detected at clicked position")
                return None, None

            # Rescale mask back to original resolution
            mask_original = rescale_labels_back(mask_isotropic, original_shape)
            mask_binary = (mask_original > 0).astype(bool)

            # Compute bounding box in original coordinates
            coords = np.argwhere(mask_binary > 0)
            if len(coords) == 0:
                return None, None

            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0) + 1

            # Crop mask to bbox region (mask must match bbox shape)
            mask_cropped = mask_binary[
                bbox_min[0] : bbox_max[0],
                bbox_min[1] : bbox_max[1],
                bbox_min[2] : bbox_max[2],
            ]

            # Flatten to [z_min, y_min, x_min, z_max, y_max, x_max]
            bbox = np.concatenate([bbox_min, bbox_max]).astype(np.int32)

            return mask_cropped, bbox

        except Exception as e:
            show_warning(f"InstanSeg inference failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, None

    def _compute_and_cache_embeddings(
        self,
        time_frame: int,
        image_isotropic: np.ndarray,
        iso_shape: tuple,
        scale_factors: tuple,
        original_shape: tuple,
    ) -> torch.Tensor:
        """Compute backbone embeddings and add to cache.

        Args:
            time_frame: Time frame index
            image_isotropic: Normalized isotropic image (Z, Y, X)
            iso_shape: Shape of isotropic volume
            scale_factors: Scale factors used for rescaling
            original_shape: Original image shape

        Returns:
            Embeddings tensor (C, D, H, W) on GPU
        """
        # Run backbone inference
        embeddings = run_backbone(self.model, image_isotropic, self.device)

        # Report GPU memory usage if available
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / 1e6
            print(f"GPU memory: {gpu_mem_mb:.0f} MB")

        # Add to cache
        self.cached_embeddings[time_frame] = (
            embeddings,
            iso_shape,
            scale_factors,
            original_shape,
        )
        self.cache_order.append(time_frame)

        # Evict oldest if cache full
        if len(self.cache_order) > self.cache_size:
            oldest_time = self.cache_order.pop(0)
            if oldest_time in self.cached_embeddings:
                del self.cached_embeddings[oldest_time]

        return embeddings

    def _convert_to_isotropic_coords(
        self, position: tuple, original_shape: tuple, iso_shape: tuple
    ) -> tuple:
        """Convert position from original to isotropic coordinates.

        Args:
            position: (z, y, x) in original coordinates
            original_shape: Original volume shape
            iso_shape: Isotropic volume shape

        Returns:
            (z_iso, y_iso, x_iso) in isotropic coordinates
        """
        scale = np.array(iso_shape) / np.array(original_shape)
        position_iso = np.array(position) * scale
        return tuple(position_iso)

    def clear_cache(self):
        """Clear all cached embeddings."""
        self.cached_embeddings.clear()
        self.cache_order.clear()
