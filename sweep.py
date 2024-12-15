import wandb
import subprocess
import os
import torch
import sys

def check_gpu_requirements():
    if not torch.cuda.is_available():
        print("Error: No CUDA device available")
        sys.exit(1)
        
    # Check memory on first GPU
    device = torch.cuda.get_device_properties(0)
    memory_gb = device.total_memory / 1024**3
    
    if memory_gb < 10:
        print(f"Error: GPU has only {memory_gb:.1f}GB memory, need at least 10GB")
        sys.exit(1)

def train_jepa(config=None):
    # Check GPU requirements first
    check_gpu_requirements()
    
    with wandb.init(config=config) as w_run:
        config = wandb.config
        
        # Get full path to main.py
        main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        
        # Ensure we have a valid run ID for experiment name
        if not wandb.run.id:
            raise ValueError("No wandb run ID available")
            
        experiment_name = f"sweep_{wandb.run.id}"
        print(f"Using experiment name: {experiment_name}")
        
        # Construct command with sweep parameters
        cmd = [
            "python", main_path,
            "--data_path", "/data/dataset/DL2572",
            "--epochs", str(config.epochs),
            "--experiment_name", experiment_name,
            "--batch_size", str(config.batch_size),
            "--seed", str(config.seed),
            "--num_kernels", str(config.num_kernels),
            "--repr_dim", str(config.repr_dim)
        ]

        print(f"Running command: {' '.join(cmd)}")
        # Run the training script
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running training command: {' '.join(cmd)}")
            print(f"Error output:\n{stderr.decode()}")
            print(f"Standard output:\n{stdout.decode()}")

# Define sweep configuration 
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'eval_normal_loss', 'goal': 'minimize'},
    'parameters': {
        'batch_size': {'values': [256, 128, 64, 512]},
        'seed': {'values': [531]}, # Multiple seeds for reproducibility
        'num_kernels': {'values': [8, 4, 16]},
        'repr_dim': {'values': [512, 256, 128, 1024]}
    }
}

def create_sweep(project_name):
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created sweep with ID: {sweep_id}")
    return sweep_id

def start_agent(project_name, sweep_id=None):
    if sweep_id is None:
        sweep_id = create_sweep(project_name)
    wandb.agent(sweep_id, train_jepa, project=project_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, help='Optional: ID of existing sweep to run')
    parser.add_argument('--project_name', type=str, default='final_jepas', help='W&B project name')
    args = parser.parse_args()
    
    # Check GPU requirements before starting
    check_gpu_requirements()
    
    start_agent(args.project_name, args.sweep_id)
