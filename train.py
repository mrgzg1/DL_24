from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import itertools
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from schedulers import Scheduler, LRSchedule
from configs import ConfigBase


class JEPAConfig(ConfigBase):
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine

default_config = JEPAConfig()

class TrainJEPA():
    def __init__(self, device, model, train_ds, val_ds, momentum, config=default_config):
        self.device = device
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = config
        self.momentum = momentum

    def train(self):
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = Scheduler(
                        schedule=self.config.schedule,
                        base_lr=self.config.lr,
                        data_loader=self.train_ds,
                        epochs=self.config.epochs,
                        optimizer=optimizer,
                        batch_size=self.train_ds.batch_size,
                    )

        for epoch in range(self.config.epochs):
            
            total_energy = 0.0
            
            with tqdm(total=len(self.train_ds), desc=f"Epoch [{epoch+1}/{self.config.epochs}]", unit="batch") as pbar:
                for batch in tqdm(self.train_ds, desc="Probe prediction step"):
                    obs = batch.states.to(self.device)
                    actions = batch.actions.to(self.device)
                    
                    optimizer.zero_grad()

                    pred_enc, tgt_enc = self.model(obs, actions, get_tgt_enc=True)

                    # Flatten the representations to combine batch and sequence dimensions
                    pred_enc_flat = pred_enc.view(-1, pred_enc.size(-1))
                    tgt_enc_flat = tgt_enc.view(-1, tgt_enc.size(-1))

                    # Compute the loss using the energy distance
                    loss = self.compute_loss(pred_enc_flat, tgt_enc_flat)

                    # Backward pass 
                    loss.backward() 
                    # Update the optimizer
                    optimizer.step()

                    # Update the target encoder
                    self.model.update_target_encoder(self.momentum)

                    total_energy += loss.item()

                    # Update the progress bar
                    pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                    pbar.update(1)

            avg_energy = total_energy / len(self.train_ds)
            print(f"Epoch [{epoch+1}/{self.config.epochs}] Average Energy Distance: {avg_energy}")


        return self.model   

    def compute_loss(self, pred_enc, tgt_enc):
        """
        Compute the energy distance between the predicted and target encodings
        """
        # Normalize the representations
        pred_enc = F.normalize(pred_enc, dim=-1)
        tgt_enc = F.normalize(tgt_enc, dim=-1)

        # Compute the energy distance
        loss = F.mse_loss(pred_enc, tgt_enc)

        return loss