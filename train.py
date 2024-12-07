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


class TrainJEPA():
    def __init__(self, device, model, train_ds, val_ds, config, save_path):
        self.device = device
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = config
        self.save_path = save_path


    def train(self):
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = Scheduler(
                        schedule=LRSchedule.Cosine,
                        base_lr=self.config.warmup_lr,
                        data_loader=self.train_ds,
                        epochs=self.config.epochs,
                        optimizer=optimizer,
                        batch_size=self.train_ds.batch_size,
                    )

        best_train_loss = float('inf')

        for epoch in range(self.config.epochs):
            
            total_energy = 0.0
            
            with tqdm(total=len(self.train_ds), desc=f"Epoch [{epoch+1}/{self.config.epochs}]", unit="batch") as pbar:
                for batch in tqdm(self.train_ds, desc="Prediction step"):
                    obs = batch.states.to(self.device)
                    actions = batch.actions.to(self.device)
                    
                    optimizer.zero_grad()

                    pred_enc, tgt_enc = self.model(obs, actions, get_tgt_enc=True)

                    # Compute the loss using the energy distance
                    loss = self.vicreg_loss(pred_enc, tgt_enc)

                    # Backward pass 
                    loss.backward() 
                    # Update the optimizer
                    optimizer.step()

                    # Update the target encoder
                    self.model.update_target_encoder(self.config.momentum)

                    total_energy += loss.item()

                    # Update the progress bar
                    pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                    pbar.update(1)

            avg_energy = total_energy / len(self.train_ds)
            print(f"Epoch [{epoch+1}/{self.config.epochs}] Average Energy Distance: {avg_energy}")

            if avg_energy < best_train_loss:
                best_train_loss = avg_energy
                self.save_model(self.save_path)



        return self.model   


    def _off_diagonal(self, matrix):
        """
        Return the off-diagonal elements of a square matrix
        """
        n, _ = matrix.shape
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, pred_enc, tgt_enc):
        """
        Compute the VicReg loss between the predicted and target encodings
        
        Args:
            pred_enc: [B, T, D]
            tgt_enc: [B, T, D]

        Output:
            loss: float
        """

        # Reshape the predicted and target encodings
        B, T, D = pred_enc.size()
        pred_enc = pred_enc.view(B*T, D)
        tgt_enc = tgt_enc.view(B*T, D)

        # Compute invarience loss
        repr_loss = F.mse_loss(pred_enc, tgt_enc)

        # Normalize by centering 
        pred_enc = pred_enc - pred_enc.mean(dim=0)
        tgt_enc = tgt_enc - tgt_enc.mean(dim=0)

        # Varience Regularization 
        std_pred = torch.sqrt(pred_enc.var(dim=0) + 1e-4)
        std_tgt =  torch.sqrt(tgt_enc.var(dim=0) + 1e-4) 
        std_loss = torch.mean(F.relu(1 - std_pred)) / 2 + torch.mean(F.relu(1 - std_tgt)) / 2

        # Covariance Regularization
        cov_pred = (pred_enc.t() @ pred_enc) / (B*T - 1) # D x D
        cov_tgt  = (tgt_enc.t() @ tgt_enc)  / (B*T - 1)  # D x D

        cov_loss = self._off_diagonal(cov_pred).pow(2).sum().div(D) + \
                   self._off_diagonal(cov_tgt).pow(2).sum().div(D)

        loss = self.config.sim_coeff * repr_loss + self.config.std_coeff * std_loss + self.config.cov_coeff * cov_loss

        return loss

    def save_model(self, path):
        path = os.path.join(path, "best_model.pth")
        torch.save(self.model.state_dict(), path)