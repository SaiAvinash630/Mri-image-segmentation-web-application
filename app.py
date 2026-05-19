"""
Stroke Detection Pipeline - Streamlit Web Application
Main UI entry point for the stroke detection pipeline.

This file contains ONLY Streamlit UI logic.
All processing is delegated to separate modules and FastAPI service.
"""

import streamlit as st
import os
import logging
import requests

from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import processing modules
from modules import convert, preprocess, volume

# ============================================================================
# Configuration
# ============================================================================

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000")

DEFAULT_PIXEL_SPACING = 0.5
DEFAULT_SLICE_THICKNESS = 2.0

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Stroke Detection Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State Initialization
# ============================================================================
debug_placeholder = st.empty()

def streamlit_debug(msg):
    debug_placeholder.write(msg)


def initialize_session_state():
    """Initialize all session state variables."""
    if "data_path" not in st.session_state:
        st.session_state.data_path = None
    
    if "converted_path" not in st.session_state:
        st.session_state.converted_path = None
    
    if "converted_files" not in st.session_state:
        st.session_state.converted_files = {}
    
    if "preprocess_values" not in st.session_state:
        st.session_state.preprocess_values = {
            "pixel_spacing": DEFAULT_PIXEL_SPACING,
            "slice_thickness": DEFAULT_SLICE_THICKNESS
        }
    
    if "preprocessed_files" not in st.session_state:
        st.session_state.preprocessed_files = {}
    
    if "mask" not in st.session_state:
        st.session_state.mask = None
    
    if "mask_path" not in st.session_state:
        st.session_state.mask_path = None
    
    if "current_slice" not in st.session_state:
        st.session_state.current_slice = 0
    
    if "volume_result" not in st.session_state:
        st.session_state.volume_result = None
    
    if "api_status" not in st.session_state:
        st.session_state.api_status = None
    if "hers_result" not in st.session_state:
        st.session_state.hers_result = None


initialize_session_state()

# ============================================================================
# Utility Functions
# ============================================================================

def check_api_status():
    """Check if FastAPI service is running."""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"API not available: {str(e)}")
        return False


def get_workflow_progress():
    """
    Get current workflow step based on session state.
    Returns step number (1-6) and whether it can proceed.
    """
    if st.session_state.data_path:
        step = 2
        if st.session_state.converted_files:
            step = 3
            if st.session_state.preprocessed_files:
                step = 4
                if st.session_state.mask_path:
                    step = 5
                    if st.session_state.volume_result:
                        step = 6
    else:
        step = 1
    
    return step


def can_proceed_to_step(target_step: int) -> bool:
    """Check if user can proceed to a specific step."""
    current_step = get_workflow_progress()
    return current_step >= target_step


# ============================================================================
# Workflow Step Functions
# ============================================================================

def step_load_data():
    """Step 1: Load data from folder OR ZIP."""
    st.subheader("📁 Step 1: Load Data")

    st.markdown("### Choose Input Method")

    option = st.radio(
        "Select input type:",
        ["📦 Upload ZIP"],
        horizontal=True
    )

    if option == "📦 Upload ZIP":
        # st.set_option('server.maxUploadSize', 1024)
        uploaded_file = st.file_uploader(
            "Upload ZIP file containing DICOM data",
            type=["zip"]
        )
        st.info("⏳ Uploading large file... please wait (this may take 2–5 minutes)")
        
        if uploaded_file is not None:

            import zipfile
            import tempfile

            # 🔍 DICOM checker (handles no-extension files)
            def is_dicom_file(file_path):
                try:
                    with open(file_path, "rb") as f:
                        f.seek(128)
                        return f.read(4) == b"DICM"
                except:
                    return False

            with st.spinner("📦 Extracting ZIP and detecting DICOM structure..."):

                try:
                    # Create temp directory
                    temp_dir = tempfile.mkdtemp()

                    zip_path = os.path.join(temp_dir, "data.zip")

                    # Save uploaded zip
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract ZIP
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # =========================================================
                    # 🔥 STRUCTURE-AWARE DETECTION (YOUR DATASET FIX)
                    # =========================================================

                    best_path = None
                    max_dcm_files = 0

                    for root, dirs, files in os.walk(temp_dir):

                        # 👉 Only consider folder named "1" or "2"
                        if os.path.basename(root) in ["1", "2"]:

                            numeric_subfolders = [
                                d for d in os.listdir(root)
                                if os.path.isdir(os.path.join(root, d)) and d.isdigit()
                            ]

                            if not numeric_subfolders:
                                continue

                            dcm_count = 0

                            # 👉 Scan inside numeric folders only
                            for sub in numeric_subfolders:
                                sub_path = os.path.join(root, sub)

                                for f in os.listdir(sub_path):
                                    file_path = os.path.join(sub_path, f)

                                    if os.path.isfile(file_path):
                                        if f.lower().endswith(".dcm") or is_dicom_file(file_path):
                                            dcm_count += 1

                            # pick best scan folder
                            if dcm_count > max_dcm_files:
                                max_dcm_files = dcm_count
                                best_path = root

                    # =========================================================
                    # RESULT
                    # =========================================================

                    if best_path and max_dcm_files > 0:
                        st.session_state.data_path = best_path

                        st.success("✅ ZIP extracted and correct DICOM dataset detected!")
                        st.write(f"📂 Using folder: {best_path}")
                        st.write(f"🧠 Total DICOM files found: {max_dcm_files}")

                        # 📋 Preview files
                        preview_files = []

                        for sub in os.listdir(best_path):
                            sub_path = os.path.join(best_path, sub)

                            if os.path.isdir(sub_path) and sub.isdigit():
                                for f in os.listdir(sub_path):
                                    preview_files.append(f)
                                    if len(preview_files) >= 20:
                                        break
                            if len(preview_files) >= 20:
                                break

                        with st.expander("📋 Sample DICOM files"):
                            for f in preview_files:
                                st.caption(f"📄 {f}")

                    else:
                        st.error("❌ Could not find valid DICOM structure (1/2 → numeric folders → files)")
                        st.session_state.data_path = None

                except Exception as e:
                    st.error(f"❌ ZIP extraction failed: {str(e)}")
                    st.session_state.data_path = None

    # =========================================================
    return st.session_state.data_path is not None


def step_convert_dicom():
    """Step 2: Convert DICOM to NIfTI."""
    st.subheader("🔄 Step 2: Convert DICOM to NIfTI")
    
    if not can_proceed_to_step(2):
        st.warning("⚠️ Please load data first (Step 1)")
        return False
    
    try:
        # Create output folder
        output_folder = os.path.join(st.session_state.data_path, "converted")
        os.makedirs(output_folder, exist_ok=True)
        
        if st.button("🔄 Convert DICOM → NIfTI", use_container_width=True, key="convert_button"):
            with st.spinner("Converting DICOM files to NIfTI format..."):
                try:
                    # Call conversion function
                    converted_files = convert.convert_dicom_to_nifti(
                        st.session_state.data_path,
                        output_folder
                    )
                    
                    # Validate output
                    is_valid, missing = convert.validate_conversion_output(converted_files)
                    
                    if is_valid:
                        st.session_state.converted_files = converted_files
                        st.session_state.converted_path = output_folder
                        st.success("✅ Conversion completed successfully!")
                        
                        with st.expander("📊 Conversion results"):
                            for modality, path in converted_files.items():
                                st.caption(f"**{modality}:** {path}")
                    else:
                        st.warning(f"⚠️ Missing modalities: {', '.join(missing)}")
                
                except Exception as e:
                    st.error(f"❌ Conversion error: {str(e)}")
                    logger.error(f"Conversion error: {str(e)}")
        
        # Show status
        if st.session_state.converted_files:
            st.success(f"✅ Converted: {', '.join(st.session_state.converted_files.keys())}")
            return True
        else:
            st.info("ℹ️ Click the button above to start conversion")
            return False
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Conversion step error: {str(e)}")
        return False


def step_preprocess():
    """Step 3: Preprocess imaging data."""
    st.subheader("⚙️ Step 3: Preprocessing")
    
    if not can_proceed_to_step(3):
        st.warning("⚠️ Please convert DICOM first (Step 2)")
        return False
    
    try:
        st.info("✨ Automatic preprocessing will:")
        st.caption("1️⃣ Run HD-BET skull stripping on all modalities")
        st.caption("2️⃣ Align ADC & FLAIR to DWI reference")
        st.caption("3️⃣ Normalize intensities")
        st.caption("4️⃣ Resize slices to 128×128")
        st.caption("5️⃣ Create model-ready .npy file")
        
        if st.button("⚙️ Start Preprocessing", use_container_width=True, key="preprocess_button"):
            with st.spinner("Preprocessing imaging data (this may take a few minutes)..."):
                try:
                    # Call preprocessing function automatically with default values
                    output_folder = os.path.join(st.session_state.converted_path, "preprocessed")
                    os.makedirs(output_folder, exist_ok=True)
                    
                    preprocessed_files = preprocess.preprocess_nifti(
                        st.session_state.converted_files,
                        output_folder,
                        do_normalize=True,
                        resample=True,
                        skull_strip=True
                    )
                    
                    st.session_state.preprocessed_files = preprocessed_files
                    st.session_state.image_path = preprocessed_files.get("NPY")
                    st.success("✅ Preprocessing completed!")
                    
                    with st.expander("📊 Preprocessing details"):
                        npy_path = preprocessed_files.get("NPY")
                        st.caption(f"**Output File:** {os.path.basename(npy_path)}")
                        st.caption(f"**HD-BET:** Applied")
                        st.caption(f"**Normalization:** Applied")
                        st.caption(f"**Target Size:** 128×128")
                
                except Exception as e:
                    st.error(f"❌ Preprocessing error: {str(e)}")
                    logger.error(f"Preprocessing error: {str(e)}")
                    return False
        
        # Show status
        if st.session_state.preprocessed_files:
            npy_path = st.session_state.preprocessed_files.get("NPY")
            if npy_path and os.path.exists(npy_path):
                st.success("✅ Preprocessing completed - Ready for prediction")
                st.info(f"📁 Preprocessed data: {os.path.basename(npy_path)}")
            else:
                st.warning("⚠️ Preprocessed file not found")
            return True
        else:
            st.info("ℹ️ Click button above to start preprocessing")
            return False
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Preprocessing step error: {str(e)}")
        return False

def step_find_mask():
    """Step 4: Find lesion mask using FastAPI."""
    st.subheader("🔍 Step 4: Find Lesion Mask (ML Prediction)")

    try:
 

        if st.session_state.get("preprocessed_files"):
            # 🟢 AUTO MODE (your main workflow)
            npy_path = st.session_state.preprocessed_files.get("NPY")

            if npy_path and os.path.exists(npy_path):
                st.success(f"✅ Using preprocessed file automatically:\n{npy_path}")
            else:
                st.error("❌ Preprocessed file missing")
                return False
        else:
                st.warning("⚠️ Please complete preprocessing first")
                return False
        # =========================================
        # 🔌 CHECK API
        # =========================================
        api_available = check_api_status()

        if not api_available:
            st.error(
                "❌ FastAPI service not running!\n\n"
                "Prediction service unavailable"
            )
            return False

        st.success("✅ FastAPI service is running")

    
        # =========================================
        # 🚀 RUN PREDICTION BUTTON (CORRECT WAY)
        # =========================================
        if st.button("🧠 Run Prediction", use_container_width=True):

            with st.spinner("Running model prediction..."):

                try:
                    npy_path = st.session_state.preprocessed_files.get("NPY")
                    
                    if not npy_path or not os.path.exists(npy_path):
                        st.error("❌ Invalid .npy file")
                        return False

                    with open(npy_path, "rb") as f:
                        files = {
                            "npy_file": ("preprocessed.npy", f, "application/octet-stream")
                        }

                        response = requests.post(
                            f"{FASTAPI_URL}/predict-preprocessed",
                            files=files,
                            timeout=900
                        )

                    if response.status_code == 200:
                        mask_save_path = os.path.join(
                            st.session_state.converted_path,
                            "preprocessed",
                            "lesion_mask.npy"
                        )

                        with open(mask_save_path, "wb") as f:
                            f.write(response.content)

                        st.session_state.mask_path = mask_save_path
                        st.success("✅ Prediction completed!")
                        st.rerun()
                    else:
                        st.error(f"❌ {response.json().get('detail')}")
                        return False

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    return False

        # =========================================
        # 🖼️ DISPLAY RESULTS
        # =========================================
        if st.session_state.get("mask_path"):

            if not os.path.exists(st.session_state.mask_path):
                st.warning("⚠️ Mask file not found")
                return False

            st.success("✅ Mask Ready")

            import numpy as np
            

            npy_path = st.session_state.preprocessed_files.get("NPY")
            data = np.load(npy_path)

            # 🔥 FIX: load NIfTI mask
            mask = np.load(st.session_state.mask_path)

            num_slices = data.shape[0]

            slice_num = st.slider("Select slice:", 0, num_slices - 1, 0)
            st.session_state.current_slice = slice_num

            dwi = data[slice_num, :, :, 0]
            adc = data[slice_num, :, :, 1]
            flair = data[slice_num, :, :, 2]
            mask_slice = mask[slice_num, :, :, 0]

            def norm(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-6)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("DWI")
                st.image(norm(dwi), clamp=True)

            with col2:
                st.subheader("ADC")
                st.image(norm(adc), clamp=True)

            with col3:
                st.subheader("FLAIR")
                st.image(norm(flair), clamp=True)

            with col4:
                st.subheader("Mask")
                st.image(mask_slice, clamp=True)

            return True

        else:
            st.info("ℹ️ Running prediction...")
            return False

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False
    



    
def step_volume_calculation():
    st.subheader("📐 Step 5: Calculate Volume")

    try:
        import nibabel as nib
        import os

        mask_path = st.session_state.get("mask_path")
        image_npy_path = (
            st.session_state.get("image_path") or
            st.session_state.get("preprocessed_files", {}).get("NPY")
        )

        if not mask_path or not os.path.exists(mask_path):
            st.warning("⚠️ Please run Step 4 first (mask not found)")
            return False

        if not image_npy_path or not os.path.exists(image_npy_path):
            st.warning("⚠️ Preprocessed image (.npy) not found")
            return False

        # =========================
        # AUTO LOAD DWI (NO INPUT)
        # =========================
        try:
            # 🔥 ALWAYS BUILD FROM DATA PATH (WORKS FOR ZIP + NORMAL)
            data_path = st.session_state.get("data_path")

            if not data_path or not os.path.exists(data_path):
                st.warning("⚠️ Data path not found")
                return False

            converted_path = os.path.join(data_path, "converted")

            if not os.path.exists(converted_path):
                st.warning("⚠️ Converted folder not found")
                return False

            dwi_path = None

            # 🔍 Find DWI automatically
            for root, dirs, files in os.walk(converted_path):
                for f in files:
                    if "dwi" in f.lower() and f.endswith((".nii", ".nii.gz")):
                        dwi_path = os.path.join(root, f)
                        break
                if dwi_path:
                    break

            if not dwi_path:
                st.error("❌ DWI file not found automatically")
                return False

            # =========================
            # LOAD USING NIBABEL
            # =========================
            img = nib.load(dwi_path)

            data = img.get_fdata()
            header = img.header

            dx, dy, dz = header.get_zooms()[:3]
            H, W, S = data.shape

            st.session_state.voxel_spacing = (dx, dy, dz)
            st.session_state.original_shape = (H, W)

            st.success("✅ 3D DWI loaded automatically")

            st.write(f"📏 Spacing: {dx:.4f}, {dy:.4f}, {dz:.4f}")
            st.write(f"📐 Shape: {(H, W)}")

        except Exception as e:
            st.error(f"❌ DWI LOAD ERROR: {str(e)}")
            return False

        # =========================
        # CALCULATE BUTTON
        # =========================
        if st.session_state.get("voxel_spacing"):

            if st.button("📐 Calculate Volume", use_container_width=True):

                result = volume.calculate_volume_from_mask(
                    mask_path=mask_path,
                    image_npy_path=image_npy_path,
                    voxel_spacing=st.session_state.voxel_spacing,
                    original_shape=st.session_state.original_shape,
                    debug_callback=st.write
                )

                st.session_state.volume_result = result
                st.success("✅ Volume calculated")

        # =========================
        # SHOW RESULT
        # =========================
        result = st.session_state.get("volume_result")

        if result:
            st.subheader("📊 Volume Results")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Volume (mm³)", result["volume_mm3"])
            col2.metric("Volume (cm³)", result["volume_cm3"])
            col3.metric("Volume (mL)", result["volume_ml"])

            severity = volume.estimate_stroke_severity(result["volume_ml"])
            col4.metric("Severity", severity)

            st.write(f"🧠 Total Volume: {result['total_volume_ml']} mL")
            st.write(f"📈 Lesion %: {result['lesion_percentage']} %")

        return True

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False



def save_slice_image(slice_array, title, path):
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        # 🔥 Fix 1: ensure numpy array
        slice_array = np.array(slice_array)

        # 🔥 Fix 2: remove NaN / Inf
        slice_array = np.nan_to_num(slice_array)

        # 🔥 Fix 3: handle empty / invalid
        if slice_array.size == 0:
            return

        # 🔥 Fix 4: normalize safely
        if np.max(slice_array) != np.min(slice_array):
            slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array))
        else:
            slice_array = np.zeros_like(slice_array)

        # 🔥 Fix 5: ensure 2D
        if slice_array.ndim > 2:
            slice_array = slice_array.squeeze()

        plt.imshow(slice_array, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"⚠️ Skipping image due to error: {e}")

def create_pdf_report(file_path, volume_result, npy_path, mask_path):

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    import numpy as np
    import os

    try:
        print("🚀 Starting PDF generation...")

        doc = SimpleDocTemplate(file_path)
        styles = getSampleStyleSheet()
        content = []

        # =========================
        # TITLE
        # =========================
        content.append(Paragraph("Stroke Detection Report", styles['Title']))
        content.append(Spacer(1, 20))

        # =========================
        # VOLUME SECTION
        # =========================
        if volume_result:
            content.append(Paragraph("Volume Analysis", styles['Heading2']))
            content.append(Spacer(1, 10))

            content.append(Paragraph(f"Lesion Volume (mL): {volume_result['volume_ml']}", styles['Normal']))
            content.append(Paragraph(f"Volume (mm³): {volume_result['volume_mm3']}", styles['Normal']))
            content.append(Paragraph(f"Volume (cm³): {volume_result['volume_cm3']}", styles['Normal']))
            content.append(Paragraph(f"Total Brain Volume (mL): {volume_result.get('total_volume_ml', 'N/A')}", styles['Normal']))
            content.append(Paragraph(f"Lesion Percentage: {volume_result['lesion_percentage']} %", styles['Normal']))

            content.append(Spacer(1, 20))

        # =========================
        # HeRS SECTION
        # =========================
        hers = volume_result.get("hers_result") if volume_result else None

        if hers:
            content.append(Paragraph("Haemorrhagic Risk Score", styles['Heading2']))
            content.append(Spacer(1, 10))

            content.append(Paragraph(f"eGFR: {hers['egfr']}", styles['Normal']))
            content.append(Paragraph(f"Category: {hers['category']}", styles['Normal']))
            content.append(Paragraph(f"Risk: {hers['risk']} %", styles['Normal']))

            content.append(Spacer(1, 20))

        # =========================
        # IMAGE SECTION
        # =========================
        data = np.load(npy_path)
        mask = np.load(mask_path)

        if mask.ndim == 4:
            mask = mask[..., 0]

        if mask.shape[0] < 10:
            mask = np.transpose(mask, (2, 0, 1))

        dwi = data[..., 0]
        adc = data[..., 1]
        flair = data[..., 2]

        os.makedirs("temp_images", exist_ok=True)

        H, W = dwi.shape[1], dwi.shape[2]
        total_pixels = H * W

        img_w = 110
        img_h = 110

        valid_count = 0

        content.append(Paragraph("Imaging Results", styles['Heading2']))
        content.append(Spacer(1, 10))

        for i in range(dwi.shape[0]):

            non_zero = np.count_nonzero(dwi[i])
            dwi_percent = (non_zero / total_pixels) * 100

            if dwi_percent <= 9:
                continue

            valid_count += 1

            content.append(Paragraph(f"Slice {i} | DWI: {dwi_percent:.2f}%", styles['Heading3']))
            content.append(Spacer(1, 10))

            dwi_path_i = f"temp_images/dwi_{i}.png"
            adc_path_i = f"temp_images/adc_{i}.png"
            flair_path_i = f"temp_images/flair_{i}.png"
            mask_path_i = f"temp_images/mask_{i}.png"

            save_slice_image(dwi[i], "DWI", dwi_path_i)
            save_slice_image(adc[i], "ADC", adc_path_i)
            save_slice_image(flair[i], "FLAIR", flair_path_i)
            save_slice_image(mask[i], "Mask", mask_path_i)

            row = [
                Image(dwi_path_i, width=img_w, height=img_h),
                Image(adc_path_i, width=img_w, height=img_h),
                Image(flair_path_i, width=img_w, height=img_h),
                Image(mask_path_i, width=img_w, height=img_h),
            ]

            table = Table([row], colWidths=[img_w]*4)
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))

            content.append(table)
            content.append(Spacer(1, 20))

        doc.build(content)

        print(f"✅ PDF CREATED with {valid_count} slices")

    except Exception as e:
        print("❌ PDF FAILED:", str(e))
        raise e
    

def step_hers():
    st.subheader("⚠️ Haemorrhagic Risk Stratification (HeRS)")

    volume_result = st.session_state.get("volume_result")
    hers_result = st.session_state.get("hers_result")

    if volume_result and hers_result:
        volume_result["hers_result"] = hers_result

    if not volume_result:
        st.warning("⚠️ Please calculate volume first")
        return False

    import math

    age = st.number_input("Age", min_value=1, max_value=120)
    creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, value=1.0)
    gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("🧠 Calculate HeRS"):

        # eGFR
        egfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
        if gender == "Female":
            egfr *= 0.742

        # category
        if egfr > 60:
            category = 0
        elif egfr >= 30:
            category = 1
        else:
            category = 2

        # logistic model
        log_odds = (
            -3.823563
            + (0.0120706 * volume_result["volume_ml"])
            + (0.5939482 * category)
            + (0.0266442 * age)
        )

        prob = 1 / (1 + math.exp(-log_odds))

        result = {
            "egfr": round(egfr, 2),
            "category": category,
            "risk": round(prob * 100, 2)
        }

        st.session_state.hers_result = result

        st.success("✅ HeRS Calculated")

        st.write(f"eGFR: {result['egfr']}")
        st.write(f"Category: {result['category']}")
        st.write(f"Risk: {result['risk']} ")

    return True



def step_download():
    """Step 6: Download results."""
    st.subheader("⬇️ Step 6: Download Results")

    # Allow access if mask exists



    try:
        download_option = st.radio(
            "What would you like to download?",
            options=["Image Mask", "Data Only", "Both"],
            horizontal=True,
            key="download_option"
        )

        file_name = st.text_input(
            "Enter file name:",
            value="stroke_report.pdf"
        )

        # =========================
        # GENERATE PDF
        # =========================
        if st.button("📄 Generate Report", use_container_width=True):

            try:
                volume_result = st.session_state.get("volume_result")
                hers_result = st.session_state.get("hers_result")
                npy_path = st.session_state.get("image_path")
                mask_path = st.session_state.get("mask_path")

                if volume_result and hers_result:
                    volume_result["hers_result"] = hers_result

                pdf_path = os.path.join(os.getcwd(), "temp_report.pdf")

                create_pdf_report(
                    pdf_path,
                    volume_result,
                    npy_path,
                    mask_path
                )

                if not os.path.exists(pdf_path):
                    st.error("❌ PDF not created")
                    return False

                st.session_state.generated_pdf = pdf_path
                st.success("✅ PDF generated!")

            except Exception as e:
                st.error(f"❌ PDF Error: {str(e)}")

        # =========================
        # DOWNLOAD BUTTON (ALWAYS VISIBLE AFTER GENERATION)
        # =========================
        if st.session_state.get("generated_pdf"):

            with open(st.session_state.generated_pdf, "rb") as f:
                st.download_button(
                    label="⬇️ Download Report",
                    data=f,
                    file_name=file_name,
                    mime="application/pdf",
                    use_container_width=True
                )

        return True

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Download step error: {str(e)}")
        return False

# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render sidebar with workflow information and controls."""
    with st.sidebar:
        st.title("⚙️ Workflow Status")
        
        # Current step indicator
        current_step = get_workflow_progress()
        st.progress(current_step / 6)
        st.caption(f"Step {current_step} of 6")
        
        # Workflow checklist
        st.subheader("📋 Workflow Progress")
        
        steps = [
            ("Load Data", st.session_state.data_path is not None),
            ("Convert DICOM", bool(st.session_state.converted_files)),
            ("Preprocess", bool(st.session_state.preprocessed_files)),
            ("Find Mask", st.session_state.mask_path is not None),
            ("Volume", st.session_state.volume_result is not None),
            ("HeRS",st.session_state.hers_result is not None),
            ("Download", st.session_state.volume_result is not None)
        ]
        
        for i, (step_name, completed) in enumerate(steps, 1):
            status = "✅" if completed else "⭕"
            st.caption(f"{status} Step {i}: {step_name}")
        
        st.divider()
        
        # Session state info
        st.subheader("📊 Session Info")
        
        if st.session_state.data_path:
            st.caption(f"**Data Path:** {st.session_state.data_path}")
        
        if st.session_state.converted_path:
            st.caption(f"**Output Path:** {st.session_state.converted_path}")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Workflow", use_container_width=True, key="reset_button"):
            for key in list(st.session_state.keys()):
                if key.startswith("data_") or key.startswith("converted_") or key.startswith("preprocess_") or key in ["mask_path", "volume_result", "current_slice"]:
                    del st.session_state[key]
            initialize_session_state()
            st.rerun()
        
        # API Status
        st.divider()
        st.subheader("🔌 API Status")
        
        if check_api_status():
            st.success("✅ FastAPI Ready")
        else:
            st.error("❌ FastAPI Not Connected")


# ============================================================================
# Main Application Layout
# ============================================================================

def main():
    """Main application layout and workflow."""
    import base64

    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # 1. Load your downloaded logo
    # Replace "logo.png" with the name of the file you downloaded

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(BASE_DIR, "logo-small.png")
    img_base64 = get_base64_of_bin_file(logo_path)

    # 2. Inject CSS and the image
    st.markdown(
        f"""
        <style>
        .top-right-logo {{
            position: absolute;
            top: -20px;
            right: 0px;
            width: 100px;
            z-index: 999;
        }}
        </style>
        <img src="data:image/png;base64,{img_base64}" class="top-right-logo">
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title("🧠 Stroke Detection Pipeline")
    st.markdown("Medical imaging analysis using deep learning for stroke lesion segmentation")
    
    st.markdown("*Developed by Manukonda Avinash (M.Tech)*\n\n---")

    # st.divider()


    # 1. Define the logo URL (or use a local path like 'logo.png')



    # Your existing header/text
    


    # Two rows of buttons
    st.subheader("🎯 Workflow Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Load Data", use_container_width=True, key="step1_btn"):
            st.session_state.active_step = 1
    
    with col2:
        if st.button("🔄 Convert", use_container_width=True, key="step2_btn"):
            st.session_state.active_step = 2
    
    with col3:
        if st.button("⚙️ Preprocess", use_container_width=True, key="step3_btn"):
            st.session_state.active_step = 3
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Find Mask", use_container_width=True, key="step4_btn"):
            st.session_state.active_step = 4

    with col2:
        if st.button("📐 Volume", use_container_width=True, key="step5_btn"):
            st.session_state.active_step = 5

    with col3:
        if st.button("⚠️ HeRS", use_container_width=True, key="step6_btn"):
            st.session_state.active_step = 6


    # 🔥 NEW ROW (Download alone)
    col1 = st.columns(1)[0]

    with col1:
        if st.button("⬇️ Download", use_container_width=True, key="step7_btn"):
            st.session_state.active_step = 7
    st.divider()
    
    # Dynamic content section
    st.subheader("📋 Workflow Content")
    
    # Get active step (default to 1)
    if "active_step" not in st.session_state:
        st.session_state.active_step = 1
    
    active_step = st.session_state.active_step
    
    # Render appropriate step
    if active_step == 1:
        step_load_data()
    elif active_step == 2:
        step_convert_dicom()
    elif active_step == 3:
        step_preprocess()
    elif active_step == 4:
        step_find_mask()
    elif active_step == 5:
        step_volume_calculation()
    elif active_step == 6:
        step_hers()
    elif active_step == 7:
        step_download()
    # Render sidebar
    render_sidebar()


if __name__ == "__main__":
    main()
