#!/bin/bash

# Ensure PROJ_NAME and SWEEP_ID are set
if [ -z "$PROJ_NAME" ] || [ -z "$SWEEP_ID" ]; then
    echo "Error: PROJ_NAME and SWEEP_ID must be set"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p resources/logs

# Run 3 agents per GPU across 8 GPUs (24 total agents)
CUDA_VISIBLE_DEVICES=0 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_0_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_0_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_0_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_1_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_1_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_1_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_2_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_2_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_2_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_3_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_3_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_3_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_4_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_4_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_4_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_5_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_5_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_5_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_6_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_6_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_6_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_7_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_7_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python sweep.py --sweep_id $SWEEP_ID > resources/logs/${SWEEP_ID}_7_3.log 2>&1 &

wait # Wait for all background processes to complete
