# Use slim Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# Create non-root user
RUN useradd -m -s /bin/bash mluser

# Install system dependencies
RUN apt-get update && apt-get upgrade && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up working directory and copy files
WORKDIR /opt/ml
COPY src/utils ./utils
COPY src/inference.py .

# Create directory for model
RUN mkdir -p /opt/ml/model && chown -R mluser:mluser /opt/ml

# Switch to non-root user
USER mluser

# Pre-download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('punkt_tab')"
# Expose port
EXPOSE ${PORT}

# Set entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "inference:app"]
