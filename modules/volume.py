import os
import logging
import numpy as np
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def calculate_volume_from_mask(
    mask_path: str,
    voxel_spacing: Tuple[float, float, float],
    original_shape: Tuple[int, int],
    image_npy_path: Optional[str] = None,
    debug_callback: Optional[callable] = None
) -> Dict[str, float]:
    """
    Calculate lesion + brain volume using mask and DWI.

    Args:
        mask_path: Path to predicted mask (.npy)
        voxel_spacing: (dx, dy, dz) in mm
        original_shape: (H, W) before resizing
        image_npy_path: preprocessed image (.npy)
        debug_callback: optional debug printer

    Returns:
        Dictionary with volume metrics
    """

    def debug(msg):
        if debug_callback:
            debug_callback(msg)
        else:
            logger.info(msg)

    # =========================
    # LOAD MASK
    # =========================
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = np.load(mask_path, mmap_mode='r')
    debug(f"Loaded mask shape: {mask.shape}")

    # Handle mask shape
    if mask.ndim == 4:
        debug("Converting 4D mask → 3D")
        mask = mask[..., 0]

    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D. Got shape: {mask.shape}")

    # Ensure shape is (S, H, W)
    if mask.shape[0] < 10:  # likely (H,W,S)
        debug("Transposing mask (H,W,S → S,H,W)")
        mask = np.transpose(mask, (2, 0, 1))

    S, H_resized, W_resized = mask.shape

    # =========================
    # LOAD IMAGE (.NPY)
    # =========================
    dwi = None
    if image_npy_path and os.path.exists(image_npy_path):
        image = np.load(image_npy_path, mmap_mode='r')
        debug(f"Loaded NPY shape: {image.shape}")

        if image.ndim != 4 or image.shape[-1] < 1:
            raise ValueError("Invalid .npy format. Expected (S,H,W,3)")

        dwi = image[..., 0]  # channel 0 = DWI
        debug(f"DWI shape: {dwi.shape}")

        if dwi.shape[0] != S:
            debug("Aligning mask with NPY slices")
            S = min(S, dwi.shape[0])
            mask = mask[:S]
            dwi = dwi[:S]

    # =========================
    # BINARIZE LESION
    # =========================
    lesion_mask = (mask > 0.5)

    # =========================
    # FILTER USING DWI
    # =========================
    if dwi is not None:

        filtered_mask = np.zeros_like(lesion_mask)

        kept_slices = 0

        for i in range(S):

            # Count non-zero pixels in DWI slice
            non_zero_pixels = np.count_nonzero(dwi[i])

            if non_zero_pixels > 9:   # 🔥 your condition (fixed)

                filtered_mask[i] = lesion_mask[i]
                kept_slices += 1

        lesion_mask = filtered_mask

        debug(f"Slices kept after DWI filter: {kept_slices}/{S}")

    total_lesion_pixels = int(np.sum(lesion_mask))
    valid_slices = int(np.sum(np.any(lesion_mask, axis=(1, 2))))

    debug(f"Lesion pixels: {total_lesion_pixels}")
    debug(f"Slices with lesion: {valid_slices}/{S}")

    # =========================
    # VOXEL SPACING
    # =========================
    dx, dy, dz = voxel_spacing
    H_orig, W_orig = original_shape

    scale_x = W_orig / W_resized
    scale_y = H_orig / H_resized

    dx_corr = dx * scale_x
    dy_corr = dy * scale_y

    voxel_volume = dx_corr * dy_corr * dz  # mm³

    debug(f"Voxel volume: {voxel_volume}")

    # =========================
    # LESION VOLUME
    # =========================
    lesion_volume_mm3 = total_lesion_pixels * voxel_volume
    lesion_volume_ml = lesion_volume_mm3 / 1000

    # =========================
    # BRAIN VOLUME (FROM DWI)
    # =========================
    brain_volume_ml = None

    if dwi is not None:
        dwi_norm = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-6)

        # simple threshold (can tune)
        brain_mask = dwi_norm > 0.1

        brain_voxels = int(np.sum(brain_mask))

        brain_volume_mm3 = brain_voxels * voxel_volume
        brain_volume_ml = brain_volume_mm3 / 1000

        debug(f"Brain voxels: {brain_voxels}")
        debug(f"Brain volume: {brain_volume_ml} mL")

    # =========================
    # FALLBACK TOTAL VOLUME
    # =========================
    total_voxels = S * H_resized * W_resized
    total_volume_ml = (total_voxels * voxel_volume) / 1000

    # =========================
    # LESION %
    # =========================
    # if brain_volume_ml and brain_volume_ml > 0:
    #     lesion_percentage = (lesion_volume_ml / brain_volume_ml) * 100
    # else:
    #     lesion_percentage = (
    #         (lesion_volume_ml / total_volume_ml) * 100
    #         if total_volume_ml > 0 else 0
    #     )
    if brain_volume_ml is None or brain_volume_ml == 0:
        raise ValueError("Brain volume is required for Option 2 calculation")

    lesion_percentage = (lesion_volume_ml / brain_volume_ml) * 100
    # =========================
    # DEBUG FINAL
    # =========================
    debug(f"Lesion volume: {lesion_volume_ml} mL")
    debug(f"Lesion %: {lesion_percentage}")

    # =========================
    # RETURN
    # =========================
    return {
        "volume_mm3": round(lesion_volume_mm3, 2),
        "volume_cm3": round(lesion_volume_ml, 4),
        "volume_ml": round(lesion_volume_ml, 4),

        "brain_volume_ml": round(brain_volume_ml, 2) if brain_volume_ml else None,

        "valid_slices": valid_slices,
        "total_lesion_pixels": total_lesion_pixels,

        "total_volume_ml": round(total_volume_ml, 2),
        "lesion_percentage": round(lesion_percentage, 4),
    }


def estimate_stroke_severity(volume_ml: float) -> str:
    if volume_ml < 10:
        return "Minor"
    elif volume_ml < 50:
        return "Moderate"
    else:
        return "Major"