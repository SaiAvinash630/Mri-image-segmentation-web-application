FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    dcm2niix \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements_app.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy project files
COPY app.py .
COPY logo-small.png .
COPY .streamlit ./.streamlit
COPY modules ./modules

# Expose Streamlit port
EXPOSE 8501

# Streamlit settings
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV FASTAPI_URL=http://api:8000

# Start app
CMD ["streamlit", "run", "app.py"]