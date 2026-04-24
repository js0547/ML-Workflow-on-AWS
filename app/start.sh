#!/bin/bash
# ------------------------------------------------------------------
# Container Entrypoint Script
# Starts both FastAPI (background) and Streamlit (foreground)
# in the same container for the Tabular ML Platform.
# ------------------------------------------------------------------

set -e

echo "============================================="
echo " Tabular ML Platform -- Starting Services"
echo "============================================="

# Start FastAPI backend in the background
echo "Starting FastAPI on port 8000..."
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1 &

FASTAPI_PID=$!
echo "FastAPI started (PID: $FASTAPI_PID)"

# Wait briefly for FastAPI to initialize
sleep 2

# Verify FastAPI is running
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "ERROR: FastAPI failed to start."
    exit 1
fi

echo "FastAPI health check..."
for i in 1 2 3 4 5; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "FastAPI is healthy."
        break
    fi
    echo "Waiting for FastAPI... (attempt $i)"
    sleep 2
done

# Start Streamlit in the foreground
echo "Starting Streamlit on port 8501..."
exec streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
