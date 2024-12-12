import wandb
import subprocess
import os

def train_jepa(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Construct command with sweep parameters
        cmd = [
            "python", "main.py",
            "--experiment_name", f"sweep_{wandb.run.id}",
            "--data_path", "/data/DL_24/data",
            "--batch_size", str(config.batch_size),
            "--learning_rate", str(config.learning_rate),
            "--epochs", str(config.epochs),
            "--encoder_type", config.encoder_type,
            "--d_model", str(config.d_model),
            "--gpu_id", "0"  # You may want to manage this differently
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
            'distribution': 'log_uniform',
            'min': math.log(1e-5),
            'max': math.log(1e-2)
        },
        'epochs': {
            'values': [10, 20, 30]
        },
        'encoder_type': {
            'values': ['cnn', 'vit']
        },
        'd_model': {
            'values': [64, 128, 256, 512]
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='wall_jepa_sweep')

# Start sweep agent
wandb.agent(sweep_id, train_jepa, count=20)  # Will run 20 different configurations