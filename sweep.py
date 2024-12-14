import wandb
import subprocess
import math
import os
import torch
import sys

def print_gpu_info():
    # Print CUDA_VISIBLE_DEVICES env var
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device.name}")
            print(f"  Total memory: {device.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {device.major}.{device.minor}")
    else:
        print("No CUDA GPUs available")

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
    
    # Print GPU info at start of training
    print("\nGPU Information:")
    print_gpu_info()
    print()
    
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
            "--experiment_name", experiment_name,
            "--data_path", "/data/dataset/DL2572",
            "--batch_size", str(config.batch_size),
            "--epochs", str(config.epochs),
            "--encoder_type", config.encoder_type,
            "--p_augment_data", str(config.p_augment_data)
        ]

        # Add optional args if explicitly set in config
        if hasattr(config, 'p_flip') and config.p_flip is not None:
            print(f"Adding p_flip: {config.p_flip}")
            cmd.extend(["--p_flip", str(config.p_flip)])
        
        if hasattr(config, 'p_noise') and config.p_noise is not None:
            print(f"Adding p_noise: {config.p_noise}")
            cmd.extend(["--p_noise", str(config.p_noise)])
            
        if hasattr(config, 'p_rotate') and config.p_rotate is not None:
            print(f"Adding p_rotate: {config.p_rotate}")
            cmd.extend(["--p_rotate", str(config.p_rotate)])

        # Add architecture args from default config
        if hasattr(config, 'num_kernels'):
            cmd.extend(["--num_kernels", str(config.num_kernels)])
            
        if hasattr(config, 'repr_dim'):
            cmd.extend(["--repr_dim", str(config.repr_dim)])
            
        if hasattr(config, 'mlp_pred_arch'):
            cmd.extend(["--mlp_pred_arch", config.mlp_pred_arch])

        cmd.extend([
            "--noise_std", str(config.noise_std),
            "--wandb_id", w_run.id
        ])
        
        print(f"Running command: {' '.join(cmd)}")
        # Run the training script
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running training command: {' '.join(cmd)}")
            print(f"Error output:\n{stderr.decode()}")
            print(f"Standard output:\n{stdout.decode()}")

# Define sweep configuration
sweep_configs = {
    # Default experiment with all augmentations enabled
    'baseline': {
        'method': 'grid',
        'metric': {'name': 'eval/combined_loss', 'goal': 'minimize'},
        'parameters': {
            'batch_size': {'values': [256]}, # from initial preference
            'epochs': {'value': 50},
            'encoder_type': {'value': 'cnn'},
            'num_kernels': {'values': [8,16]}, # narrower range based on initial insights
            'repr_dim': {'values': [512]},
            'mlp_pred_arch': {'value': '1024-512-256'}, # pick the one that seemed better initially
            'p_augment_data': {'values': [0]},
        }
    },
    # Experiment with only flip augmentation
    'flip_only': {
        'method': 'grid',
        'metric': {'name': 'eval/combined_loss', 'goal': 'minimize'},
        'parameters': {
            'batch_size': {'values': [256]},
            'epochs': {'value': 50},
            'encoder_type': {'value': 'cnn'},
            'num_kernels': {'values': [8,16]},
            'repr_dim': {'values': [512]},
            'mlp_pred_arch': {'value': '1024-512-256'},
            'p_augment_data': {'values': [0.2, 0.4, 0.7]},
            'p_flip': {'value': 1.0},
            'p_noise': {'value': 0},
            'p_rotate': {'value': 0},
        }
    },
    # Experiment with only noise augmentation
    'noise_only': {
        'method': 'grid',
        'metric': {'name': 'eval/combined_loss', 'goal': 'minimize'},
        'parameters': {
            'batch_size': {'values': [256]},
            'epochs': {'value': 50},
            'encoder_type': {'value': 'cnn'},
            'num_kernels': {'values': [8,16]},
            'repr_dim': {'values': [512]},
            'mlp_pred_arch': {'value': '1024-512-256'},
            'p_augment_data': {'values': [0.2, 0.4, 0.7]},
            'p_flip': {'value': 0},
            'p_noise': {'value': 1.0},
            'p_rotate': {'value': 0},
            'noise_std': {'values': [0.01,0.02,0.05]},
        }
    },
    # Experiment with only rotation augmentation
    'rotate_only': {
        'method': 'grid',
        'metric': {'name': 'eval/combined_loss', 'goal': 'minimize'},
        'parameters': {
            'batch_size': {'values': [256]},
            'epochs': {'value': 50},
            'encoder_type': {'value': 'cnn'},
            'num_kernels': {'values': [8,16]},
            'repr_dim': {'values': [512]},
            'mlp_pred_arch': {'value': '1024-512-256'},
            'p_augment_data': {'values': [0.2, 0.4, 0.7]},
            'p_flip': {'value': 0},
            'p_noise': {'value': 0},
            'p_rotate': {'value': 1.0},
        }
    },
 }

def create_sweep(project_name, experiment_type):
    if experiment_type not in sweep_configs:
        raise ValueError(f"Invalid experiment type. Choose from: {list(sweep_configs.keys())}")
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_configs[experiment_type], project=project_name)
    print(f"Created sweep for {experiment_type} with ID:")
    print(sweep_id)
    return sweep_id

def start_agent(project_name, experiment_type, sweep_id=None):
    if sweep_id is None:
        sweep_id = create_sweep(project_name, experiment_type)
    
    # Start sweep agent
    wandb.agent(sweep_id, train_jepa, project=project_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, help='Optional: ID of existing sweep to run')
    parser.add_argument('--project_name', type=str, default='wall_jepa_sweep', help='W&B project name')
    parser.add_argument('--experiment_type', type=str, default="default",
                      choices=['flip_only', 'rotate_only', 'noise_only', 'baseline'],
                      help='Type of experiment to run')
    args = parser.parse_args()
    
    # Check GPU requirements before starting
    check_gpu_requirements()
    
    # Print GPU info before starting agent
    print("\nAvailable GPU Information:")
    print_gpu_info()
    print()
    
    start_agent(args.project_name, args.experiment_type, args.sweep_id)