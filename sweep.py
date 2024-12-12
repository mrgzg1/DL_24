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
            "--learning_rate", str(config.learning_rate),
            "--epochs", str(config.epochs),
            "--encoder_type", config.encoder_type,
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
            'values': [32, 64, 128, 256, 512]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.999
        },
        'epochs': {
            'values': [10, 20, 30]
        },
        'encoder_type': {
            'values': ['cnn']
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='wall_jepa_sweep')
print("Agent:")
print(sweep_id)

# Start sweep agent
wandb.agent(sweep_id, train_jepa, count=20)  # Will run 20 different configurations