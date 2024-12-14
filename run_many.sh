#!/bin/bash

# Number of agents to run per GPU
NUM_PER_GPU=2

# Ensure SWEEP_ID and PROJ_NAME are set
if [ -z "$SWEEP_ID" ] || [ -z "$PROJ_NAME" ]; then
    echo "Error: SWEEP_ID and PROJ_NAME must be set"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p resources/logs

# Run agents across GPUs using the provided sweep ID
for AGENT in $(seq 1 $NUM_PER_GPU); do
    for GPU in {1..7}; do
        CUDA_VISIBLE_DEVICES=$GPU python sweep.py --sweep_id $SWEEP_ID --project_name $PROJ_NAME > resources/logs/${SWEEP_ID}_${GPU}_${AGENT}.log 2>&1 &
        echo "Started agent $AGENT on GPU $GPU, sleeping for 10"
        sleep 10
    done
done

wait # Wait for all background processes to complete
