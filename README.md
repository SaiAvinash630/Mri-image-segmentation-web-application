# MRI Image Segmentation Web Application

[![GitHub repo size](https://img.shields.io/github/repo-size/SaiAvinash630/Mri-image-segmentation-web-application)](https://github.com/SaiAvinash630/Mri-image-segmentation-web-application)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Streamlit-based MRI image segmentation web application with a FastAPI backend for model prediction.

This repository contains:

- `app.py` - Streamlit UI for upload, preprocessing, volume estimation, and report generation
- `modules/` - preprocessing and volume calculation helper modules
- `modules_service/` - FastAPI inference service and prediction model loader
- `requirements_app.txt` - Python dependencies for the Streamlit app
- `modules_service/requirements_api.txt` - Python dependencies for the FastAPI service
- `Dockerfile` and `modules_service/Dockerfile` - container images for the app and API service

> Note: The model weight file `modules_service/model_epoch213_trainDice0.9208.keras` is excluded from the repository via `.gitignore`.

## Requirements

- Python 3.11
- pip
- A local or remote model artifact at `modules_service/model_epoch213_trainDice0.9208.keras`

## Local Setup

1. Clone the repo or use the existing project directory.

2. Create and activate a virtual environment:

```powershell
cd G:\web_application
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the Streamlit app dependencies:

```powershell
pip install -r requirements_app.txt
```

4. Install the FastAPI service dependencies:

```powershell
pip install -r modules_service/requirements_api.txt
```

5. Ensure the model file is available:

```powershell
# Place or download the model artifact here
modules_service\model_epoch213_trainDice0.9208.keras
```

## Run the API Service

Start the FastAPI model service first:

```powershell
cd modules_service
uvicorn api:app --host 0.0.0.0 --port 8000
```

The service health endpoint will be available at `http://127.0.0.1:8000/health`.

## Run the Streamlit App

In a separate terminal, from the project root:

```powershell
cd G:\web_application
$env:FASTAPI_URL = "http://127.0.0.1:8000"
streamlit run app.py
```

Open the Streamlit UI in your browser at the URL shown in the terminal (typically `http://localhost:8501`).

## Docker Compose

You can run both services together using Docker Compose.

1. Ensure the model file exists at `modules_service/model_epoch213_trainDice0.9208.keras`.
2. Run:

```powershell
docker-compose up --build
```

3. The FastAPI service will be available at `http://localhost:8000` and the Streamlit app at `http://localhost:8501`.

4. To stop the services, use:

```powershell
docker-compose down
```

## Notes

- The app expects the backend API to be available at `FASTAPI_URL`.
- If you run the API on a different host or port, update the `FASTAPI_URL` environment variable before starting Streamlit.
- The repository does not currently include a `docker-compose.yml`, so Docker support is available only via the individual Dockerfiles.

## Project Structure

- `app.py` - main Streamlit application
- `modules/convert.py` - DICOM conversion utilities
- `modules/preprocess.py` - image preprocessing utilities
- `modules/volume.py` - volume calculation utilities
- `modules_service/api.py` - FastAPI endpoints
- `modules_service/predict.py` - model wrapper and prediction logic
- `Dockerfile` - Streamlit app image
- `modules_service/Dockerfile` - FastAPI service image

## Troubleshooting

- If the API fails to start, verify the model file exists at `modules_service/model_epoch213_trainDice0.9208.keras`.
- If Streamlit cannot reach the API, confirm `FASTAPI_URL` is set correctly and the API container/service is running.
