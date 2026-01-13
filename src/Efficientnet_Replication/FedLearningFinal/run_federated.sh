#!/bin/bash

# Start server in background
python federated_local.py server &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start all clients in background
for i in {0..4}; do
    python federated_local.py client $i &
done

# Wait for all processes
wait