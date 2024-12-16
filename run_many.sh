#!/bin/bash

# Check if sweep_id is provided
if [ $# -eq 0 ]; then
	    echo "Please provide a sweep_id as an argument."
	        echo "Usage: $0 <sweep_id>"
		    exit 1
fi

# Get the sweep_id from the first argument
sweep_id=$1

# Run sweep.py on GPUs 1-7
for gpu_id in {1..7}
do
	    export CUDA_VISIBLE_DEVICES=$gpu_id
	        python sweep.py --sweep_id $sweep_id >> resources/gpu${gpu_id}.log 2>&1 &
		    echo "Started sweep on GPU $gpu_id" 
		    sleep 10
	    done

	    # Wait for all background processes to finish
	    wait

	    echo "All sweep processes have completed."
