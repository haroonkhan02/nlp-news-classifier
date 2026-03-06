# Multi-stage build for production BERT News Classifier
# Stage 1: Build dependencies
# Stage 2: Slim runtime image

# ---- Build stage ----
FROM python:3.10-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints /app/mlruns

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV MODEL_PATH=/app/checkpoints/bert-news-best.pt
ENV MODEL_NAME=bert-base-uncased
ENV MAX_LEN=256
ENV DEVICE=cpu
ENV PYTHONPATH=/app

# Health check for Kubernetes liveness probe
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with uvicorn (2 workers for CPU, scale with K8s replicas)
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
