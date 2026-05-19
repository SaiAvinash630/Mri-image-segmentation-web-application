FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application files
COPY api.py .
COPY predict.py .
COPY model_epoch213_trainDice0.9208.keras .

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]