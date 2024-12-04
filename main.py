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


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    model = JEPA(device="cpu")
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

def train_jepa(device, model, train_ds, val_ds, momentum, best_model_path):
    trainer = TrainJEPA(device=device, model=model, train_ds=train_ds, val_ds=val_ds, momentum=momentum, save_path=best_model_path)
    model = trainer.train()

def save_model(model, path):
    torch.save(model.state_dict(), path)




if __name__ == "__main__":

    # Set global paths
    folder_path = "/".join(sys.path[0].split("/")[:]) + "/"
    best_model_path = folder_path + "best_models/"

    # Create model path if it does not exist
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    # evaluate_model(device, model, probe_train_ds, probe_val_ds)
    train_jepa(device, model, probe_train_ds, probe_val_ds, momentum=0.99, best_model_path=best_model_path)
