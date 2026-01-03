#!/bin/bash
# Start the server in background
python services/icd10_dbsearch.py > data/logs/server.log 2>&1 &

# Save PID if you want to kill later
SERVER_PID=$!

# Start the client in foreground
python UI.py

# When client exits, kill server
kill $SERVER_PID