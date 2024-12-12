#!/bin/bash

# Ensure PROJ_NAME and SWEEP_ID are set
if [ -z "$PROJ_NAME" ] || [ -z "$SWEEP_ID" ]; then
    echo "Error: PROJ_NAME and SWEEP_ID must be set"
    exit 1
fi

# Run 3 agents per GPU across 8 GPUs (24 total agents)
CUDA_VISIBLE_DEVICES=0 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=0 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=0 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=1 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=1 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=1 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=2 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=2 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=2 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=3 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=3 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=3 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=4 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=4 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=4 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=5 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=5 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=5 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=6 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=6 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=6 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

CUDA_VISIBLE_DEVICES=7 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=7 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &
CUDA_VISIBLE_DEVICES=7 python run_agent.py --sweep_id $SWEEP_ID --project $PROJ_NAME &

wait # Wait for all background processes to complete
