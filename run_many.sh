#!/bin/bash

# Number of agents to run per GPU
NUM_PER_GPU=2

# Ensure PROJ_NAME and experiment type are set
if [ -z "$PROJ_NAME" ] || [ -z "$EXP_TYPE" ]; then
    echo "Error: PROJ_NAME and EXP_TYPE must be set"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p resources/logs

# Start initial sweep process in background and capture its log file
INIT_LOG="resources/logs/init_sweep.log"
python sweep.py --project_name $PROJ_NAME --experiment_type $EXP_TYPE > $INIT_LOG 2>&1 &
INIT_PID=$!
echo "$INIT_PID, $INIT_LOG"
# Wait up to 60 seconds and monitor log for sweep ID
SWEEP_ID=""
COUNTER=0
while [ $COUNTER -lt 10 ]; do
    if [ -f $INIT_LOG ]; then
        SWEEP_ID=$(grep -o 'Created sweep for .* with ID:.*' $INIT_LOG | awk '{print $NF}')
	if [ ! -z "$SWEEP_ID" ]; then
            echo "Captured sweep ID: $SWEEP_ID"
            break
        fi
    fi
    sleep 1
    ((COUNTER++))
done

# Kill initial process since we have the sweep ID
kill $INIT_PID 2>/dev/null
echo "Init PID: $INIT_PID was killed"

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to get sweep ID within 60 seconds"
    exit 1
fi

# Run agents across GPUs using the captured sweep ID
for AGENT in $(seq 1 $NUM_PER_GPU); do
    for GPU in {1..7}; do
        CUDA_VISIBLE_DEVICES=$GPU python sweep.py --sweep_id $SWEEP_ID --project_name $PROJ_NAME > resources/logs/${SWEEP_ID}_${GPU}_${AGENT}.log 2>&1 &
        echo "Started agent $AGENT on GPU $GPU, sleeping for 10"
        sleep 10
    done
done

wait # Wait for all background processes to complete
