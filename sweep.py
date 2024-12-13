import wandb
import subprocess
import math
import os

def train_jepa(config=None):
    with wandb.init(config=config) as w_run:
        config = wandb.config
        
        # Get full path to main.py
        main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        # Construct command with sweep parameters
        cmd = [
            "python", main_path,
            "--experiment_name", f"sweep_{wandb.run.id}",
            "--data_path", "/data/dataset/DL2572",
            "--batch_size", str(config.batch_size),
            "--epochs", str(config.epochs),
            "--encoder_type", config.encoder_type,
            "--p_augment_data", str(config.p_augment_data)
        ]

        # Only add optional args if explicitly set in config
        if hasattr(config, 'p_flip') and config.p_flip is not None:
            cmd.extend(["--p_flip", str(config.p_flip)])
        
        if hasattr(config, 'p_noise') and config.p_noise is not None:
            cmd.extend(["--p_noise", str(config.p_noise)])
            
        if hasattr(config, 'p_rotate') and config.rotate is not None:
            cmd.extend(["--p_rotate", str(config.rotate)])

        cmd.extend([
            "--noise_std", str(config.noise_std),
            "--wandb_id", w_run.id
        ])
        
        print(f"running: {cmd}")
        # Run the training script
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running training: {stderr.decode()}")

# Define sweep configuration
sweep_configs = {
    # Experiment Set 1: Only flips
    'flip_only': {
        'method': 'grid',
        'metric': {
            'name': 'eval/combined_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'value': 256},
            'epochs': {'value': 30},
            'encoder_type': {'value': 'cnn'},
            'p_augment_data': {'values': [0.01, 0.05, 0.1, 0.3]},
            'p_flip': {'value': 1.0},
            'p_noise': {'value': 0},
            'p_rotate': {'value': 0},
            'noise_std': {'value': 0.05}
        }
    },
    
    # Experiment Set 2: Only rotations
    'rotate_only': {
        'method': 'grid',
        'metric': {
            'name': 'eval/combined_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'value': 256},
            'epochs': {'value': 30},
            'encoder_type': {'value': 'cnn'},
            'p_augment_data': {'values': [0.01, 0.05, 0.1, 0.3]},
            'p_flip': {'value': 0},
            'p_noise': {'value': 0},
            'p_rotate': {'value': 1.0},
            'noise_std': {'value': 0.05}
        }
    },
    
    # Experiment Set 3: Only noise with varying std
    'noise_only': {
        'method': 'grid',
        'metric': {
            'name': 'eval/combined_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'value': 256},
            'epochs': {'value': 30},
            'encoder_type': {'value': 'cnn'},
            'p_augment_data': {'values': [0.01, 0.05, 0.1, 0.3]},
            'p_flip': {'value': 0},
            'p_noise': {'value': 1.0},
            'p_rotate': {'value': 0},
            'noise_std': {'values': [0.001, 0.005, 0.01, 0.05]}
        }
    },
    
    # Experiment Set 4: All augmentations together
    'all_augs': {
        'method': 'grid',
        'metric': {
            'name': 'eval/combined_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'value': 256},
            'epochs': {'value': 30},
            'encoder_type': {'value': 'cnn'},
            'p_augment_data': {'values': [0.01, 0.05, 0.1, 0.3]},
            'p_flip': {'value': None},
            'p_noise': {'value': None},
            'p_rotate': {'value': None},
            'noise_std': {'value': 0.05}
        }
    }
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
    parser.add_argument('--experiment_type', type=str, required=True, 
                      choices=['flip_only', 'rotate_only', 'noise_only', 'all_augs'],
                      help='Type of experiment to run')
    args = parser.parse_args()
    
    start_agent(args.project_name, args.experiment_type, args.sweep_id)