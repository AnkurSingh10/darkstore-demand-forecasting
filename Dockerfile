FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system build dependencies required for C++ libraries (XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, model artifacts, and processed data only
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Expose Streamlit default port
EXPOSE 8501

# Health check for Streamlit service
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit dashboard on container startup
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
