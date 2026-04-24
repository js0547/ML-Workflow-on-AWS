# ------------------------------------------------------------------
# Dockerfile for the Tabular ML Platform
# Runs both Streamlit (UI) and FastAPI (backend) in a single container.
# Designed for ECS Fargate deployment.
# ------------------------------------------------------------------

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# Copy and install application dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Copy pipeline code (needed for pipeline.py imports and SageMaker SDK source_dir)
COPY pipeline/ /app/pipeline/

# Copy the entrypoint script and make it executable
COPY app/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create a Streamlit config to disable telemetry and set server options
RUN mkdir -p /root/.streamlit
RUN echo '[general]\n\
email = ""\n\
\n\
[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n' > /root/.streamlit/config.toml

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Health check -- verify Streamlit is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the entrypoint script
ENTRYPOINT ["/app/start.sh"]
