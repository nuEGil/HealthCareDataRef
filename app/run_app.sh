#!/bin/bash
set -e

# Kill any existing services on the ports before starting
echo "Cleaning up existing services..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true  # icd10_dbsearch port
lsof -ti:8001 | xargs kill -9 2>/dev/null || true  # knowledge_store_search port
lsof -ti:8002 | xargs kill -9 2>/dev/null || true  # image_model_server port

sleep 1

echo "Cleaned services"
# Start all services in background
python -m app.services.icd10_dbsearch > app/data/logs/icd_10_db_serve.log 2>&1 &
PID_ICD=$!

python -m app.services.knowledge_store_search > app/data/logs/knowledge_db_serve.log 2>&1 &
PID_KNOW=$!

python -m app.services.image_model_server > app/data/logs/image_model_server.log 2>&1 &
PID_IMAGE=$!

# Give services a moment to start up
sleep 2

# Start the UI in foreground
python -m app.UI

# Cleanup function
cleanup(){
    echo "Shutting down services..."
    kill $PID_ICD $PID_KNOW $PID_IMAGE 2>/dev/null
}

trap cleanup EXIT