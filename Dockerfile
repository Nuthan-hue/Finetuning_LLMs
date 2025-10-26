# Multi-stage build for Kaggle Multi-Agent System
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage (includes dev tools)
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data models submissions logs notebooks

# Expose ports
EXPOSE 8000 8888

# Production stage (minimal)
FROM base as production

# Copy only necessary files
COPY src/ /app/src/
COPY .env.example /app/.env.example

# Create necessary directories
RUN mkdir -p data models submissions logs

# Create non-root user for security
RUN useradd -m -u 1000 kaggle && \
    chown -R kaggle:kaggle /app

USER kaggle

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/main.py"]

# Use development stage by default
FROM development