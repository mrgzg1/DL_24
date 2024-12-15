from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
from train import TrainJEPA
import torch
import numpy as np
import random
# from models.prober import MockModel
from models.jepa import JEPA
import glob
import sys
import time
import os
import wandb
from configs import parse_args, save_args, check_folder_paths, load_args

CONFIG = None # used as global to save args

def get_device(args):
    """Set CUDA device based on gpu_id and return device."""
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Using device: cuda:{args.gpu_id}")
    return device


def load_data_probe(device, batch_size):
    data_path = CONFIG.data_path
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
        batch_size=batch_size,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
        batch_size=batch_size,
    )

    probe_val_wall_other_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall_other/val",
        probing=True,
        device=device,
        train=False,
        batch_size=batch_size,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
        "wall_other": probe_val_wall_other_ds,
    }

    return probe_train_ds, probe_val_ds


def load_expert_data(device, batch_size):
    data_path = CONFIG.data_path

    probe_train_expert_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_expert/train",
        probing=True,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    probe_val_expert_ds = {
        "expert": create_wall_dataloader(
            data_path=f"{data_path}/probe_expert/val",
            probing=True,
            device=device,
            train=False,
            batch_size=batch_size,
        )
    }

    return probe_train_expert_ds, probe_val_expert_ds


def load_data_jepa(device, batch_size):
    data_path = CONFIG.data_path

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    return train_ds


def load_model(device, config):
    """Load or initialize the model."""
    model = JEPA(device=device, config=config)
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")
        wandb.log({f"eval_{probe_attr}_loss": loss})

def train_jepa(device, model, train_ds, probe_train_ds, probe_val_ds, probe_train_expert_ds, probe_val_expert_ds, config, save_path):
    trainer = TrainJEPA(
        device=device, 
        model=model, 
        train_ds=train_ds, 
        config=config, 
        save_path=save_path,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        probe_train_expert_ds=probe_train_expert_ds,
        probe_val_expert_ds=probe_val_expert_ds
    )
    model = trainer.train()

def load_model_weights(model, path, device):
    # Load model weights and move to device
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.device = device
    return model

if __name__ == "__main__":
    args = parse_args()
    CONFIG = args

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize wandb
    wandb.init(
        project="wall_jepa", 
        name=args.experiment_name,
        config=args
    )

    folder_path = "/".join(sys.path[0].split("/")[:]) + "/"
    resources_path = os.path.join(folder_path, "resources")
    experiment_path = os.path.join(resources_path, "experiments", args.experiment_name)
    device = get_device(args)

    # train if eval isn't set
    if not args.eval:
        # Check if experiment directory already exists
        if os.path.exists(experiment_path):
            response = input(f"Warning: Experiment directory {experiment_path} already exists. Continue and overwrite? [y/N] ")
            if response.lower() != 'y':
                print("Aborting...")
                sys.exit(0)
                
        check_folder_paths([os.path.join(experiment_path, "checkpoints")])
        # save arguments
        save_args(args, os.path.join(experiment_path, "args.txt"))

        model_path = os.path.join(experiment_path, "checkpoints")
        train_ds = load_data_jepa(device, args.batch_size)
        probe_train_ds, probe_val_ds = load_data_probe(device, args.batch_size)
        probe_train_expert_ds, probe_val_expert_ds = load_expert_data(device, args.batch_size)
        model = load_model(device, args)
        train_jepa(
            device, 
            model, 
            train_ds, 
            probe_train_ds, 
            probe_val_ds,
            probe_train_expert_ds,
            probe_val_expert_ds,
            config=args, 
            save_path=model_path
        )
    else:
        # evaluate the model at the end of every run anyways
        if not os.path.exists(experiment_path):
            print(f"Error: Experiment directory {experiment_path} does not exist")
            sys.exit(1)
            
        # Fix path construction to ensure proper joining
        checkpoints_dir = os.path.join(experiment_path, "checkpoints")
        # checkpoint_files = ['best_model.pth'] # uncomment me to load best_mode.pth
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    
        
        if not checkpoint_files:
            print(f"Error: No checkpoint files found in {checkpoints_dir}")
            sys.exit(1)
            
        print(f"Experiment path: {experiment_path}")
        print("Found checkpoints:", checkpoint_files)

        # Use the same device as specified in args
        device = get_device(args)
        probe_train_ds, probe_val_ds = load_data_probe(device, args.batch_size)
        probe_train_expert_ds, probe_val_expert_ds = load_expert_data(device, args.batch_size)
        model = load_model(device, args)

        # Evaluate each checkpoint
        for checkpoint_path in checkpoint_files:
            print("\nTesting JEPA model:", checkpoint_path)
            try:
                model = load_model_weights(model, checkpoint_path, device)
                evaluate_model(device, model, probe_train_ds, probe_val_ds)
                evaluate_model(device, model, probe_train_expert_ds, probe_val_expert_ds)
            except Exception as e:
                print(e)
                print(f"Error loading/evaluating checkpoint {checkpoint_path}: {str(e)}")
                raise(e)
                continue
