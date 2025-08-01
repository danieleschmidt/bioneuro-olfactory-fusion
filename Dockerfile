# Multi-stage Docker build for BioNeuro-Olfactory-Fusion

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml README.md LICENSE ./
COPY bioneuro_olfactory/ ./bioneuro_olfactory/

# Build wheel
RUN pip install build && \
    python -m build --wheel

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r biouser && useradd -r -g biouser biouser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install application
RUN pip install --no-cache-dir /tmp/*.whl[sensors] && \
    rm /tmp/*.whl

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R biouser:biouser /app

# Switch to non-root user
USER biouser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import bioneuro_olfactory; print('OK')" || exit 1

# Default command
CMD ["bioneuro-monitor", "--help"]

# Labels
LABEL maintainer="daniel@terragonlabs.com"
LABEL description="Neuromorphic gas detection system"
LABEL version="0.1.0"