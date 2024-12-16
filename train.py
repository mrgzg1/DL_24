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
import wandb

from schedulers import Scheduler, LRSchedule
from configs import ConfigBase
from evaluator import ProbingEvaluator


class TrainJEPA():
    def __init__(self, device, model, train_ds, config, save_path, probe_train_ds=None, probe_val_ds=None, probe_train_expert_ds=None, probe_val_expert_ds=None):
        self.device = device
        self.model = model
        self.train_ds = train_ds
        self.config = config
        self.save_path = save_path
        
        # Initialize evaluator if probe datasets are provided
        if probe_train_ds is not None and probe_val_ds is not None:
            self.evaluator = ProbingEvaluator(
                device=device,
                model=model,
                probe_train_ds=probe_train_ds,
                probe_val_ds=probe_val_ds,
                quick_debug=False
            )
            
        if probe_train_expert_ds is not None and probe_val_expert_ds is not None:
            self.expert_evaluator = ProbingEvaluator(
                device=device,
                model=model,
                probe_train_ds=probe_train_expert_ds,
                probe_val_ds=probe_val_expert_ds,
                quick_debug=False
            )

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
        step = 0
        lr = self.config.warmup_lr  # Initialize learning rate

        for epoch in range(self.config.epochs):
            
            total_energy = 0.0
            
            with tqdm(total=len(self.train_ds), desc=f"Epoch [{epoch+1}/{self.config.epochs}]", unit="batch") as pbar:
                for batch in tqdm(self.train_ds, desc="Prediction step"):
                    obs = batch.states.to(self.device)
                    actions = batch.actions.to(self.device)
                    
                    optimizer.zero_grad()

                    pred_enc, tgt_enc = self.model(obs, actions, get_tgt_enc=True)

                    # Compute the loss using the energy distance
                    if self.config.loss_type == 'vicreg':
                        loss = self.vicreg_loss(pred_enc, tgt_enc, step)

                    elif self.config.loss_type == 'byol':
                        loss = self.byol_loss(pred_enc, tgt_enc, step)

                    # Backward pass 
                    loss.backward() 
                    # Update the optimizer
                    optimizer.step()

                    # Update the target encoder
                    self.model.update_target_encoder()

                    total_energy += loss.item()

                    # Update the progress bar
                    pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                    pbar.update(1)

                    # Log metrics to wandb
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': lr
                    }, step=step)

                    # Update the learning rate
                    lr = scheduler.adjust_learning_rate(step)
                    step += 1

            avg_energy = total_energy / len(self.train_ds)
            print(f"Epoch [{epoch+1}/{self.config.epochs}] Average Energy Distance: {avg_energy}")

            # Evaluate model after each epoch if evaluator exists
            if hasattr(self, 'evaluator'):
                print("Evaluating model on normal data...")
                prober = self.evaluator.train_pred_prober()
                avg_losses = self.evaluator.evaluate_all(prober=prober)
                print(f"wall: {avg_losses['wall']} | normal: {avg_losses['normal']}")
                
                # Log evaluation metrics
                wandb.log({
                    'epoch': epoch,
                    'avg_energy': avg_energy,
                    'eval/normal_loss': avg_losses['normal'],
                    'eval/wall_loss': avg_losses['wall'],
                    'eval/combined_loss': (avg_losses['normal'] + avg_losses['wall'])/2
                }, step=step)

            if hasattr(self, 'expert_evaluator'):
                print("Evaluating model on expert data...")
                expert_prober = self.expert_evaluator.train_pred_prober()
                expert_losses = self.expert_evaluator.evaluate_all(prober=expert_prober)
                print(f"expert loss: {expert_losses['expert']}")
                
                # Log expert evaluation metrics
                wandb.log({
                    'eval/expert_loss': expert_losses['expert']
                }, step=step)

            if avg_energy < best_train_loss:
                best_train_loss = avg_energy
                self.save_model(self.save_path)
                wandb.run.summary['best_train_loss'] = best_train_loss

            if epoch % 5 == 0:
                self.save_model(self.save_path, int(epoch/5))

        return self.model   

    def _off_diagonal(self, matrix):
        """
        Return the off-diagonal elements of a square matrix
        """
        n, _ = matrix.shape
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, pred_enc, tgt_enc, step):
        """
        Compute the VicReg loss for each time step and accumulate over the temporal dimension.

        Args:
            pred_enc: [B, T, D]
            tgt_enc: [B, T, D]
            step: Current training step for logging

        Output:
            loss: float
        """

        # Retrieve dimensions
        B, T, D = pred_enc.size()
        
        # Initialize loss accumulators
        repr_loss_accum = 0
        std_loss_accum = 0
        cov_loss_accum = 0

        # Iterate over time steps
        for t in range(T):
            pred_step = pred_enc[:, t, :]  # [B, D]
            tgt_step = tgt_enc[:, t, :].detach()  # [B, D], detach target for no gradients

            # Normalize representations
            # pred_step = F.normalize(pred_step, dim=-1, p=2)
            # tgt_step = F.normalize(tgt_step, dim=-1, p=2)

            # Invariance loss
            repr_loss = F.mse_loss(pred_step, tgt_step)

            # Center the embeddings
            pred_step = pred_step - pred_step.mean(dim=0)
            tgt_step = tgt_step - tgt_step.mean(dim=0)

            # Variance regularization
            std_pred = torch.sqrt(pred_step.var(dim=0) + 1e-4)
            std_tgt = torch.sqrt(tgt_step.var(dim=0) + 1e-4)
            std_loss = torch.mean(F.relu(1 - std_pred)) / 2 + torch.mean(F.relu(1 - std_tgt)) / 2

            # Covariance regularization
            cov_pred = (pred_step.t() @ pred_step) / (B - 1)  # D x D
            cov_tgt = (tgt_step.t() @ tgt_step) / (B - 1)  # D x D

            cov_loss = self._off_diagonal(cov_pred).pow(2).sum().div(D) + \
                    self._off_diagonal(cov_tgt).pow(2).sum().div(D)

            # Accumulate losses
            repr_loss_accum += repr_loss
            std_loss_accum += std_loss
            cov_loss_accum += cov_loss

        # Average losses across all time steps
        repr_loss_accum /= T
        std_loss_accum /= T
        cov_loss_accum /= T

        # Final loss with coefficients
        loss = (self.config.sim_coeff * repr_loss_accum +
                self.config.std_coeff * std_loss_accum +
                self.config.cov_coeff * cov_loss_accum)

        # Log VicReg components
        wandb.log({
            'vicreg/repr_loss': repr_loss_accum.item(),
            'vicreg/std_loss': std_loss_accum.item(), 
            'vicreg/cov_loss': cov_loss_accum.item(),
            'vicreg/total_loss': loss.item()
        }, step=step)

        return loss

    ######### BYOL Loss #########

    def byol_loss(self, pred_enc, tgt_enc, step):
        """
        Compute the BYOL loss between the predicted and target encodings for each time step.
        
        Args:
            pred_enc: [B, T, D] - Output from predictor network
            tgt_enc: [B, T, D] - Output from target network (detached)
            step: Current training step for logging

        Output:
            loss: float - BYOL loss value
        """
        B, T, D = pred_enc.size()
        total_loss = 0.0

        for t in range(T):
            # Extract embeddings at time step t
            pred_enc_t = pred_enc[:, t, :]  # [B, D]
            tgt_enc_t = tgt_enc[:, t, :]    # [B, D]

            # Normalize the representations
            pred_enc_t = F.normalize(pred_enc_t, dim=-1, p=2)  # [B, D]
            tgt_enc_t = F.normalize(tgt_enc_t, dim=-1, p=2)    # [B, D]

            # Compute BYOL loss for the current time step
            loss_t = -2 * (pred_enc_t * tgt_enc_t.detach()).sum(dim=-1).mean()

            # Accumulate loss
            total_loss += loss_t

        # Average loss over all time steps
        total_loss = total_loss / T

        # Log BYOL loss
        wandb.log({
            'byol/total_loss': total_loss.item()
        }, step=step)

        return total_loss

    def save_model(self, path, i=None):
        if i is not None:
            save_path = os.path.join(path, f"best_model_{i}.pth")
            artifact_name = f"model-checkpoint-{i}"
        else:
            save_path = os.path.join(path, "best_model.pth")
            artifact_name = "best-model"
            
        # Save model locally
        torch.save(self.model.state_dict(), save_path)
        if i is None:
            # Log model artifact to wandb
            artifact = wandb.Artifact(
                artifact_name,
                type="model",
                description=f"Model checkpoint at iteration {i if i is not None else 'best'}"
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)