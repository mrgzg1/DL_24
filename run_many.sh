# set SWEEP_ID and run this script to take over all the available GPU, estimated 10G / agent / GPU
CUDA_VISIBLE_DEVICES=1 wandb agent -p wall_jepa_sweep $SWEEP_ID


