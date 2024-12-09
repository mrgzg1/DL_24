"""
TODO: Implement JEPA model

Input:
- states: [B, T, Ch, H, W]
- actions: [B, T-1, 2]

Output:
- predictions: [B, T, D]

Where:
- B: batch size
- T: number of timesteps
- Ch: number of channels
- H: height
- W: width
- D: representation dimension

The model should have the following architecture:
- Given an agent trajectory 𝜏 , i.e. an observation-action sequence
- Recurrent JEPA consists of a RESNET-18 backbone encoder, Recurrent Predictor, and target encoder
- The RESNET-18 backbone encoder processes the first frame of the trajectory to produce a feature representation
- The Recurrent Predictor LSTM processes the remaining frames of the trajectory to predict the future feature representation at each timestep using the previous feature representation and action
- The target encoder processes the target trajectory to produce a feature representation at each timestep
- The objective is to minimize the energy distance between the predicted and target feature representations at each timestep
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List
from torchvision.models import resnet18
from .encoder import ResNetEncoder, CNNBackbone
from .predictor import Predictor

class JEPA(nn.Module):
    def __init__(self, device="cpu", bs=64, n_steps=17, enc_dim=256, action_dim=2, config=None):
        super(JEPA, self).__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256
        self.config = config
        self.encoder = CNNBackbone(n_kernels=32, repr_dim=enc_dim)
        self.target_encoder = self.get_target_encoder()
        self.predictor = Predictor(enc_dim=enc_dim, action_dim=action_dim, arch="1024-1024-1024", n_steps=n_steps)

        
    def get_target_encoder(self):

        target_encoder = copy.deepcopy(self.encoder)

        for param in target_encoder.parameters():
            param.requires_grad = False

        return target_encoder

    # def _init_target_encoder(self):
    #     # Initializing target encoder parameters with the same values as the encoder
    #     for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
    #         target_param.data.copy_(param.data)

    def update_target_encoder(self, momentum=0.99):
        # Update target encoder parameters with the encoder parameters using momentum
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * param.data

    def forward(self, obs, actions, get_tgt_enc=False):
        """
        Args:
            obs: [B, T, Ch, H, W]
            actions: [B, T-1, 2]
            get_tgt_enc: bool

        Output:
            pred_enc: [B, T, D]
            tgt_enc: [B, T, D]

        """

        predicted_encodings = []
        target_encodings = []

        # Encoding the first frame of the trajectory
        first_frame = obs[:, 0, :, :, :]
        state_t = self.encoder(first_frame)
        predicted_encodings.append(state_t.unsqueeze(1))

        if get_tgt_enc:

            with torch.no_grad():
                target_encoding = self.target_encoder(first_frame)
            target_encodings.append(target_encoding.unsqueeze(1))

        # Predicting the future feature representation at each timestep
        for t in range(1, self.n_steps):
            action = actions[:, t-1, :] # Get the action at timestep t-1

            state_t = self.predictor(state_t, action) # Predict the future feature representation at timestep t
            predicted_encodings.append(state_t.unsqueeze(1))

            if get_tgt_enc:
                with torch.no_grad():
                    target_encoding = self.target_encoder(obs[:, t, :, :, :])
                target_encodings.append(target_encoding.unsqueeze(1))

        # Stack the predicted feature representations for each timestep
        predicted_encodings = torch.cat(predicted_encodings, dim=1)

        if get_tgt_enc:
            target_encodings = torch.cat(target_encodings, dim=1)
            return predicted_encodings, target_encodings

        return predicted_encodings

if __name__ == "__main__":

    # Test JEPA
    model = JEPA(device="cpu")
    obs = torch.randn((64, 17, 2, 65, 65))
    actions = torch.randn((64, 16, 2))
    pred_enc = model(obs, actions)
    print(pred_enc.size())
    # print(pred_enc)

    # Test JEPA with loss
    model = JEPA(device="cpu")
    obs = torch.randn((64, 17, 2, 65, 65))
    actions = torch.randn((64, 16, 2))
    pred_enc, loss = model(obs, actions, compute_loss=True)
    print(pred_enc.size())
    # print(pred_enc)
    print(loss)



    

    

        




    






