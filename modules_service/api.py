"""
FastAPI Service for Model Prediction
Provides REST API endpoint for stroke detection predictions.
"""

import os
import logging

from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
# Import the prediction module from same directory
from predict import StrokeDetectionModel
from fastapi import FastAPI, HTTPException



print("🚀 RUNNING API FROM:", os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stroke Detection API",
    description="API for stroke lesion segmentation using deep learning",
    version="1.0.0"
)

# Global model instance
model = None
# MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_epoch213_trainDice0.9208.keras")
MODEL_DIR = BASE_DIR

@app.on_event("startup")
async def load_model_on_startup():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        logger.info(f"🔥 Loading model from: {MODEL_PATH}")
        model = StrokeDetectionModel(MODEL_PATH)
        logger.info(f"Checking model path exists: {os.path.exists(MODEL_PATH)}")
        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


@app.on_event("shutdown")
async def cleanup_on_shutdown():
    """Cleanup when the API shuts down."""
    logger.info("API shutting down...")


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    dwi_path: str
    adc_path: str
    flair_path: str
    output_path: Optional[str] = None



class PredictionResponse(BaseModel):
    """Response model for successful prediction."""
    success: bool
    message: str
    output_path: Optional[str] = None
    mask_shape: Optional[tuple] = None




class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    is_loaded: bool
    model_path: Optional[str] = None
    message: str


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and model.is_loaded if model else False
    }


@app.get("/model-status", response_model=ModelStatusResponse, tags=["Model"])
async def get_model_status():
    """Get current model status."""
    if model is None:
        return ModelStatusResponse(
            is_loaded=False,
            model_path=None,
            message="No model instance initialized"
        )
    
    return ModelStatusResponse(
        is_loaded=model.is_loaded,
        model_path=model.model_path,
        message="Model loaded" if model.is_loaded else "Model not loaded"
    )


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_stroke_lesion(request: PredictionRequest):

    # Check model first
    global model
    if model is None or not model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. API startup may have failed."
        )

    try:
        # Validate input files exist
        for path, name in [
            (request.dwi_path, "DWI"),
            (request.adc_path, "ADC"),
            (request.flair_path, "FLAIR")
        ]:
            if not os.path.exists(path):
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} file not found: {path}"
                )

        # Run prediction
        logger.info("Running prediction...")
        mask = model.predict(
            request.dwi_path,
            request.adc_path,
            request.flair_path,
            request.output_path
        )

        return PredictionResponse(
            success=True,
            message="Prediction completed successfully",
            output_path=request.output_path,
            mask_shape=tuple(mask.shape)
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )




from fastapi import UploadFile, File
import tempfile
import numpy as np

@app.post("/predict-preprocessed", response_model=PredictionResponse, tags=["Prediction"])
async def predict_stroke_lesion_preprocessed(
    npy_file: UploadFile = File(...)
):
    temp_path = None

    try:
        global model

        if model is None or not model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            contents = await npy_file.read()
            tmp.write(contents)
            temp_path = tmp.name

        preprocessed_data = np.load(temp_path)

        logger.info(f"Loaded npy: shape={preprocessed_data.shape}")

        output_path = temp_path.replace(".npy", "_mask.npy")

        mask = model.predict_preprocessed(
            preprocessed_data,
            output_path
        )

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename="lesion_mask.npy"
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )




@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(
    data_list: list,
    output_dir: Optional[str] = None
):

    
    if model is None or not model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        results = model.batch_predict(data_list, output_dir)
        
        successful = sum(1 for v in results.values() if v is not None)
        
        return {
            "success": True,
            "results": {k: (tuple(v.shape) if v is not None else None) for k, v in results.items()},
            "successful_predictions": successful,
            "total_predictions": len(data_list),
            "output_dir": output_dir
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/info", tags=["Info"])
async def get_info():
    """Get API information."""
    return {
        "name": "Stroke Detection API",
        "version": "1.0.0",
        "model_directory": MODEL_DIR,
        "endpoints": {
            "health": "/health",
            "model_status": "/model-status",
            "predict": "/predict",
            "predict_batch": "/predict-batch"
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stroke Detection API",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
