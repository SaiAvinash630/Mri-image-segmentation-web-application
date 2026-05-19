import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional
import nibabel as nib

# =========================
# SETTINGS (CHANGE THIS)
# =========================
DCM2NIIX = "dcm2niix"

# =========================
# MAIN FUNCTION
# =========================
def convert_dicom_to_nifti(
    input_path: str,
    output_folder: str,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Dict[str, Optional[str]]:

    OUTPUT_ROOT = Path(output_folder)
    TEMP_ROOT = OUTPUT_ROOT / "_tmp_convert"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    # =========================
    # FIND LEAF FOLDERS
    # =========================
    def find_leaf_folders(root_dir):
        leafs = []
        for root, dirs, files in os.walk(root_dir):
            if files and len(dirs) == 0:
                leafs.append(root)
        return leafs

    leafs = find_leaf_folders(input_path)
    total = len(leafs)

    # =========================
    # CONVERT DICOM → NIFTI
    # =========================
    for i, folder in enumerate(leafs):

        if progress_callback:
            progress_callback((i + 1) / total)

        cmd = [
            DCM2NIIX,
            "-z", "y",
            "-m", "n",
            "-ba", "n",
            "-f", "%p_%s",
            "-o", str(TEMP_ROOT),
            folder
        ]

        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        print("\n==== DCM2NIIX OUTPUT ====")
        print(result.stdout)
        print("==== ERRORS ====")
        print(result.stderr)

    # =========================
    # CLASSIFICATION LOGIC
    # =========================
    def classify(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_type = str(data.get("ImageType", "")).upper()
        series_desc = str(data.get("SeriesDescription", "")).upper()
        protocol = str(data.get("ProtocolName", "")).upper()

        bval = data.get("B_value", None)
        try:
            bval = float(bval)
        except:
            bval = None

        # ---- DWI / ADC ----
        if "DIFFUSION" in image_type:
            if "ADC" in image_type:
                return "ADC"
            if bval is None or bval >= 800:
                return "DWI"
            else:
                return "DWI_LOW"

        # ---- FLAIR ----
        if (
            "FLAIR" in series_desc
            or "FLAIR" in protocol
            or "T2_FLAIR" in series_desc
            or "DARK" in series_desc
            or "t2_tse_dark-fluid_tra" in series_desc
        ):
            return "FLAIR"

        return "OTHER"

    # =========================
    # MOVE FILES INTO FOLDERS
    # =========================
    for jf in os.listdir(TEMP_ROOT):
        if not jf.endswith(".json"):
            continue

        json_path = TEMP_ROOT / jf

        try:
            label = classify(json_path)

            if label not in ["DWI", "ADC", "FLAIR"]:
                continue

            out_folder = OUTPUT_ROOT / label
            out_folder.mkdir(exist_ok=True)

            base = jf[:-5]

            nii_gz = TEMP_ROOT / (base + ".nii.gz")
            nii = TEMP_ROOT / (base + ".nii")

            shutil.move(json_path, out_folder / jf)

            if nii_gz.exists():
                shutil.move(nii_gz, out_folder / (base + ".nii.gz"))
            elif nii.exists():
                shutil.move(nii, out_folder / (base + ".nii"))

        except Exception as e:
            print("Error:", e)

    # =========================
    # SPLIT 4D DWI → 3D
    # =========================
    def split_4d(file_path):
        img = nib.load(file_path)
        data = img.get_fdata()

        if len(data.shape) == 4:
            for i in range(data.shape[3]):
                vol = data[:, :, :, i]
                new_img = nib.Nifti1Image(vol, img.affine, img.header)

                new_name = file_path.replace(".nii.gz", f"_vol{i:02d}.nii.gz")
                nib.save(new_img, new_name)

            os.remove(file_path)

    dwi_folder = OUTPUT_ROOT / "DWI"
    if dwi_folder.exists():
        for f in os.listdir(dwi_folder):
            if f.endswith(".nii.gz"):
                split_4d(str(dwi_folder / f))

    # =========================
    # SELECT REQUIRED FILES
    # =========================
    result = {
        "DWI": None,
        "ADC": None,
        "FLAIR": None
    }

    # ---- ADC ----
    adc_folder = OUTPUT_ROOT / "ADC"
    if adc_folder.exists():
        for f in os.listdir(adc_folder):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                result["ADC"] = str(adc_folder / f)
                break

    # ---- FLAIR ----
    flair_folder = OUTPUT_ROOT / "FLAIR"
    if flair_folder.exists():
        for f in os.listdir(flair_folder):
            if "tra" in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz")):
                result["FLAIR"] = str(flair_folder / f)
                break

    # ---- DWI ----
    if dwi_folder.exists():
        for f in os.listdir(dwi_folder):
            if "vol01" in f.lower():
                result["DWI"] = str(dwi_folder / f)
                break

    # =========================
    # CLEAN TEMP FILES
    # =========================
    try:
        shutil.rmtree(TEMP_ROOT)
    except:
        pass

    return result


# =========================
# VALIDATION FUNCTION
# =========================
def validate_conversion_output(converted_files: dict, required_modalities=None):

    if required_modalities is None:
        required_modalities = ["DWI", "ADC", "FLAIR"]

    missing = []

    for modality in required_modalities:
        if (
            modality not in converted_files
            or not converted_files[modality]
            or not Path(converted_files[modality]).exists()
        ):
            missing.append(modality)

    return len(missing) == 0, missing