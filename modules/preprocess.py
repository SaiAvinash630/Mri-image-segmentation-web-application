"""
Preprocessing Module
Handles normalization, resampling, and other preprocessing steps.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# =========================
# SETTINGS
# =========================
# 🔥 HD-BET VENV PYTHON
HD_BET_COMMAND = "hd-bet"


TARGET_SIZE = (128, 128)


def validate_spacing_values(
    pixel_spacing: float,
    slice_thickness: float
) -> Tuple[bool, Optional[str]]:
    """
    Validate pixel spacing and slice thickness values.

    Args:
        pixel_spacing: Pixel spacing in mm (should be > 0)
        slice_thickness: Slice thickness in mm (should be > 0)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if pixel_spacing <= 0:
        return False, "Pixel spacing must be greater than 0"
    if slice_thickness <= 0:
        return False, "Slice thickness must be greater than 0"
    if pixel_spacing > 10:  # Sanity check
        return False, "Pixel spacing seems too large (typical range: 0.5-5 mm)"
    if slice_thickness > 10:  # Sanity check
        return False, "Slice thickness seems too large (typical range: 1-5 mm)"

    return True, None


def run_hdbet(input_path: str, output_path: str) -> str:
    """
    Run HD-BET using its CLI from a separate venv
    """

    # Path to Scripts folder of venv
    hdbet_exe = HD_BET_COMMAND

    command = [
        hdbet_exe,
        "-i", input_path,
        "-o", output_path,
        "-device", "cpu"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("\n====== HD-BET OUTPUT ======")
    print(result.stdout)
    print("====== HD-BET ERROR ======")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"HD-BET failed:\n{result.stderr}")

    # Handle naming issue
    if os.path.exists(output_path):
        return output_path

    alt_output = input_path.replace(".nii", "_brain.nii.gz")
    if os.path.exists(alt_output):
        return alt_output

    raise FileNotFoundError("HD-BET output not found")


def load_nifti(path: str) -> Tuple[sitk.Image, np.ndarray]:
    """
    Load NIfTI file using SimpleITK.

    Args:
        path: Path to NIfTI file

    Returns:
        Tuple of (sitk_image, numpy_array)
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    # Handle 4D volumes (take first volume)
    if arr.ndim == 4:
        arr = arr[0]

    return img, arr.astype(np.float32)


def resample_to_reference(moving_img: sitk.Image, reference_img: sitk.Image) -> np.ndarray:
    """
    Resample moving image to match reference image spacing/orientation.

    Args:
        moving_img: Image to resample
        reference_img: Reference image

    Returns:
        Resampled numpy array
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    return sitk.GetArrayFromImage(resampler.Execute(moving_img))


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize image intensities to [0, 1] range.

    Args:
        img: Input image array

    Returns:
        Normalized image array
    """
    if np.max(img) == 0:
        return img
    return img / (np.max(img) + 1e-8)


def preprocess_nifti(
    input_files: Dict[str, str],
    output_folder: str,
    pixel_spacing: float = 1.0,
    slice_thickness: float = 1.0,
    do_normalize: bool = True,
    resample: bool = True,
    skull_strip: bool = True
) -> Dict[str, str]:
    """
    Comprehensive preprocessing pipeline for stroke detection.
    Automatically processes DWI, ADC, FLAIR from converted files.

    Args:
        input_files: Dictionary from convert.convert_dicom_to_nifti()
        output_folder: Path where preprocessed data will be saved
        pixel_spacing: Pixel spacing in mm (auto-detected if possible)
        slice_thickness: Slice thickness in mm (auto-detected if possible)
        normalize: Whether to normalize intensity values
        resample: Whether to resample to standard spacing
        skull_strip: Whether to perform skull stripping

    Returns:
        Dictionary with 'NPY' key pointing to preprocessed numpy array file
    """
    # Extract paths from converted files
    dwi_path = input_files.get("DWI")
    adc_path = input_files.get("ADC")
    flair_path = input_files.get("FLAIR")

    logger.info("=== PREPROCESSING START ===")
    logger.info(f"DWI:   {dwi_path}")
    logger.info(f"ADC:   {adc_path}")
    logger.info(f"FLAIR: {flair_path}")

    if not (dwi_path and adc_path and flair_path):
        raise ValueError(f"❌ Missing required files. Got: {input_files}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Output paths for skull-stripped files
    dwi_out = os.path.join(output_folder, "DWI_brain.nii.gz")
    adc_out = os.path.join(output_folder, "ADC_brain.nii.gz")
    flair_out = os.path.join(output_folder, "FLAIR_brain.nii.gz")

    # ===========================
    # RUN HD-BET
    # ===========================
    if skull_strip:
        logger.info("\n🧠 Running HD-BET skull stripping...")

        try:
            dwi_out = run_hdbet(dwi_path, dwi_out)
            adc_out = run_hdbet(adc_path, adc_out)
            flair_out = run_hdbet(flair_path, flair_out)
            logger.info("✅ HD-BET completed")
        except Exception as e:
            logger.error(f"❌ HD-BET failed: {str(e)}")
            raise

    # ===========================
    # LOAD IMAGES
    # ===========================
    logger.info("\n📥 Loading and processing images...")
    dwi_img, dwi = load_nifti(dwi_out)
    adc_img, adc = load_nifti(adc_out)
    flair_img, flair = load_nifti(flair_out)

    # ===========================
    # ALIGN TO DWI
    # ===========================
    logger.info("🔗 Aligning ADC and FLAIR to DWI reference...")
    adc = resample_to_reference(adc_img, dwi_img)
    flair = resample_to_reference(flair_img, dwi_img)

    # ===========================
    # NORMALIZE
    # ===========================
    if do_normalize:
        logger.info("📊 Normalizing intensities...")
        dwi = normalize(dwi)
        adc = normalize(adc)
        flair = normalize(flair)

    # ===========================
    # SLICE PROCESSING
    # ===========================
    logger.info("✂️ Processing slices to 128×128...")
    min_slices = min(dwi.shape[0], adc.shape[0], flair.shape[0])

    X_data = []

    for i in range(min_slices):
        # Resize to 128×128
        dwi_slice = cv2.resize(dwi[i], TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        adc_slice = cv2.resize(adc[i], TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        flair_slice = cv2.resize(flair[i], TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # Stack: (DWI, ADC, FLAIR)
        img = np.stack([dwi_slice, adc_slice, flair_slice], axis=-1)

        if img.shape == (128, 128, 3):
            X_data.append(img)

    X_data = np.array(X_data, dtype=np.float32)

    logger.info(f"📦 Processed {len(X_data)} slices to shape {X_data.shape}")

    # ===========================
    # SAVE
    # ===========================
    npy_path = os.path.join(output_folder, "preprocessed.npy")
    np.save(npy_path, X_data)

    logger.info(f"\n✅ PREPROCESSING COMPLETE")
    logger.info(f"💾 Saved: {npy_path}")
    logger.info(f"📐 Shape: {X_data.shape}")
    logger.info("=" * 50)

    return {"NPY": npy_path}