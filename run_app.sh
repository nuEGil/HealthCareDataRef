#!/bin/bash
set -e
# Start the server in background

python services/icd10_dbsearch.py > data/logs/icd_10_db_serve.log 2>&1 &

# Save PID if you want to kill later
PID_ICD=$!

python services/knowledge_store_search.py > data/logs/knowledge_db_serve.log 2>&1 &
PID_KNOW=$!
# Start the client in foreground
python UI.py

cleanup(){
    kill $PID_ICD $PID_KNOW 2>/dev/null
}
trap cleanup EXIT



