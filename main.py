from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
from train import TrainJEPA
import torch
# from models.prober import MockModel
from models.jepa import JEPA
import glob
import sys
import time
import os
from configs import parse_args, save_args, check_folder_paths, load_args



TRAIN_JEPA = False


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data_probe(device, batch_size):
    data_path = "/home/pratyaksh/arpl/workspaces/ws_dynamics/jepa_2d_simulation/data/DL24FA"

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

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_data_jepa(device, batch_size):
    
    data_path = "/home/pratyaksh/arpl/workspaces/ws_dynamics/jepa_2d_simulation/data/DL24FA"
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
    # TODO: Replace MockModel with your trained model
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

def train_jepa(device, model, train_ds, config, save_path):
    trainer = TrainJEPA(device=device, model=model, train_ds=train_ds, config=config, save_path=save_path)
    model = trainer.train()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model_weights(model, path, device):
    # Load model weights and move to device
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.device = device
    return model

if __name__ == "__main__":

    if TRAIN_JEPA:

        # parse arguments
        args = parse_args()

        # Set global paths
        folder_path = "/".join(sys.path[0].split("/")[:]) + "/"
        resources_path = folder_path + "resources/"
        experiment_path = resources_path + "experiments/" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(args.run_id) + "/"

        check_folder_paths([os.path.join(experiment_path, "checkpoints")])
        # save arguments
        save_args(args, os.path.join(experiment_path, "args.txt"))

        model_path = experiment_path + "checkpoints/"
        device = get_device()
        train_ds = load_data_jepa(device, args.batch_size)
        model = load_model(device, args)
        train_jepa(device, model, train_ds, config=args, save_path=model_path)

    else:
        
        # parse arguments
        args = parse_args()

        folder_path = "/".join(sys.path[0].split("/")[:]) + "/"
        resources_path = folder_path + "resources/"

        experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
        model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)

        print(experiment_path)
        print("Testing JEPA model:", model_path)
        args = load_args(experiment_path + "args.txt")


        device = get_device()
        probe_train_ds, probe_val_ds = load_data_probe(device, args.batch_size)
        model = load_model(device, args)
        
        model = load_model_weights(model, model_path, device)
        evaluate_model(device, model, probe_train_ds, probe_val_ds)
