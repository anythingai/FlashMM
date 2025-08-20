# Multi-stage Dockerfile for FlashMM
# Optimized for production deployment with security best practices

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.5.1
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --only main --no-dev

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN groupadd -r flashmm && useradd -r -g flashmm flashmm

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R flashmm:flashmm /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY --chown=flashmm:flashmm research/models/exported/ ./models/ 2>/dev/null || mkdir -p ./models/

# Set ownership
RUN chown -R flashmm:flashmm /app

# Switch to non-root user
USER flashmm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "flashmm.main"]