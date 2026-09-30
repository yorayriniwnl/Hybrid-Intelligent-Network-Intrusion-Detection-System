# Multi-stage production container for Hybrid Intelligent NIDS
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install CPU PyTorch first to keep image lightweight, followed by project dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, artifacts, and backend
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY artifacts/ /app/artifacts/

# Expose standard FastAPI application port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch production server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
