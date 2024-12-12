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
            "--data_path", "/data/DL_24/data", 
            "--batch_size", str(config.batch_size),
            "--epochs", str(config.epochs),
            "--encoder_type", config.encoder_type,
            "--p_augment_data", str(config.p_augment_data),
            "--wandb_id", w_run.id
        ]
        
        # Run the training script
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running training: {stderr.decode()}")

# Define sweep configuration  
sweep_config = {
    'method': 'bayes',  # Using Bayesian optimization
    'metric': {
        'name': 'eval/combined_loss',  # The metric we want to optimize
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [256, 512, 1024]
        },
        'epochs': {
            'values': [20]
        },
        'encoder_type': {
            'values': ['cnn']
        },
        'p_augment_data': {
            'values': [0, 0.01, 0.025, 0.05, 0.1]
        }
    }
}

def create_sweep(project_name):
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print("Created sweep with ID:")
    print(sweep_id)
    return sweep_id

def start_agent(project_name, sweep_id=None):
    if sweep_id is None:
        sweep_id = create_sweep(project_name)
    
    # Start sweep agent
    wandb.agent(sweep_id, train_jepa, count=20, project=project_name)  # Will run 20 different configurations

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, help='Optional: ID of existing sweep to run')
    parser.add_argument('--project_name', type=str, default='wall_jepa_sweep', help='W&B project name')
    args = parser.parse_args()
    
    start_agent(args.project_name, args.sweep_id)