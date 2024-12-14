from typing import NamedTuple, Optional
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


def horizontal_flip(image, action):
    """Apply horizontal flip augmentation to image and adjust action accordingly"""
    aug_image = torch.flip(image, dims=[3])  # Flip both channels horizontally
    aug_action = action.clone()
    aug_action[:, 0] = -aug_action[:, 0]  # Reverse x-component of actions
    return aug_image, aug_action

def vertical_flip(image, action):
    """Apply vertical flip augmentation to image and adjust action accordingly"""
    aug_image = torch.flip(image, dims=[2])  # Flip both channels vertically
    aug_action = action.clone()
    aug_action[:, 1] = -aug_action[:, 1]  # Reverse y-component of actions
    return aug_image, aug_action

def rotate_90(image, action):
    """
    Rotate image by 90 degrees clockwise and adjust action vector accordingly
    
    Args:
        image: tensor of shape (B, C, H, W)
        action: tensor of shape (B, 2) containing (dx, dy) vectors
    """
    aug_image = torch.rot90(image, k=1, dims=[2, 3])  # Rotate both channels clockwise
    aug_action = action.clone()
    
    # For 90 degree clockwise rotation:
    # x' = y
    # y' = -x
    old_x = aug_action[:, 0].clone()
    aug_action[:, 0] = aug_action[:, 1]
    aug_action[:, 1] = -old_x
    
    return aug_image, aug_action

def add_gaussian_noise(image, std=0.05):
    """Add Gaussian noise to the wall channel only (channel=1), for all time steps."""
    # image has shape (T, C, H, W)
    # Select the wall channel across all time steps: aug_image[:, 1, ...]
    aug_image = image.clone()
    noise = torch.randn_like(aug_image[:, 1]) * std  # Noise same shape as wall channel across T
    aug_image[:, 1] = torch.clamp(aug_image[:, 1] + noise, 0, 1)
    return aug_image



def apply_augmentations(image, action, p_aug=0.5, p_hflip=None, p_vflip=None, p_rot90=None, p_noise=None, noise_std=0.05):
    """Apply all augmentations with given probabilities for the entire trajectory"""
    aug_image = image
    aug_action = action
    
    # Set individual probabilities to overall p_aug if not specified
    p_hflip = p_hflip if p_hflip is not None else p_aug
    p_vflip = p_vflip if p_vflip is not None else p_aug
    p_rot90 = p_rot90 if p_rot90 is not None else p_aug
    p_noise = p_noise if p_noise is not None else p_aug
    
    # Only apply augmentations with probability p_aug
    if torch.rand(1).item() < p_aug:
        # Horizontal flip
        if torch.rand(1).item() < p_hflip:
            aug_image, aug_action = horizontal_flip(aug_image, aug_action)
            
        # Vertical flip    
        if torch.rand(1).item() < p_vflip:
            aug_image, aug_action = vertical_flip(aug_image, aug_action)
            
        # 90 degree rotation
        if torch.rand(1).item() < p_rot90:
            aug_image, aug_action = rotate_90(aug_image, aug_action)
            
        # Add noise
        if torch.rand(1).item() < p_noise:
            aug_image = add_gaussian_noise(aug_image, std=noise_std)
    
    return aug_image, aug_action
class WallDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        p_aug=0.0,
        p_flip=None,
        p_rotate=None,
        p_noise=None,
        noise_std=0.05,
        cache_size=128,  # Number of samples to cache in GPU memory
    ):
        self.device = device
        self.cache_size = cache_size  # Number of samples to cache
        self.data_cache = {}  # Cached data storage
        self.cache_start = 0  # Start index of the current cache

        # Use memory mapping to handle large datasets
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy", mmap_mode="r")

        self.locations = None
        if probing:
            try:
                self.locations = np.load(f"{data_path}/locations.npy", mmap_mode="r")
            except FileNotFoundError:
                print("Warning: locations.npy not found, skipping location loading.")

        self.apply_augs = p_aug > 0
        self.p_aug = p_aug
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise
        self.noise_std = noise_std

        self._print_data_stats()

    def __len__(self):
        return len(self.states)

    def _load_to_cache(self, start_idx):
        """Load a subset of data into GPU memory."""
        end_idx = min(start_idx + self.cache_size, len(self.states))
        states = torch.from_numpy(self.states[start_idx:end_idx]).float()
        actions = torch.from_numpy(self.actions[start_idx:end_idx]).float()

        locations = (
            torch.from_numpy(self.locations[start_idx:end_idx]).float()
            if self.locations is not None
            else None
        )

        # Move to GPU if device is CUDA
        self.data_cache = {
            "states": states.to(self.device),
            "actions": actions.to(self.device),
            "locations": locations.to(self.device) if locations is not None else None,
        }
        self.cache_start = start_idx

    def __getitem__(self, i):
        # Ensure the requested index is in the cache
        if not (self.cache_start <= i < self.cache_start + self.cache_size):
            self._load_to_cache(i)

        cache_idx = i - self.cache_start
        states = self.data_cache["states"][cache_idx]
        actions = self.data_cache["actions"][cache_idx]
        locations = (
            self.data_cache["locations"][cache_idx]
            if self.data_cache["locations"] is not None
            else torch.empty(0, device=self.device)
        )

        # Apply augmentations if required
        if self.apply_augs:
            states, actions = apply_augmentations(
                states,
                actions,
                p_aug=self.p_aug,
                p_hflip=self.p_flip,
                p_vflip=self.p_flip,
                p_rot90=self.p_rotate,
                p_noise=self.p_noise,
                noise_std=self.noise_std,
            )

        return WallSample(states=states, locations=locations, actions=actions)

    def _print_data_stats(self):
        print("Data statistics:")
        print(f"States: {self.states.shape}")
        print(f"Actions: {self.actions.shape}")

        if self.locations is not None:
            print(f"Locations: {self.locations.shape}")

        print("\nAugmentation parameters:")
        print(f"Apply augmentations: {self.apply_augs}")
        if self.apply_augs:
            print(f"p_aug: {self.p_aug}")
            print(f"p_flip: {self.p_flip}")
            print(f"p_rotate: {self.p_rotate}")
            print(f"p_noise: {self.p_noise}")
            print(f"noise_std: {self.noise_std}")
        print()

    def display_trajectory(self, i):
        """Visualize a trajectory of observations."""
        states = torch.from_numpy(self.states[i]).cpu().numpy()
        actions = torch.from_numpy(self.actions[i]).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title("Trajectory of the agent")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        for t in range(states.shape[0]):
            state = states[t]

            # Display the wall image
            wall_image = state[0]
            ax.imshow(wall_image, cmap="gray")

            # Overlay the agent image
            agent_image = state[1]
            ax.imshow(agent_image, cmap="jet", alpha=0.5)

            plt.pause(0.5)
            plt.draw()

        plt.show()
        plt.close()

def create_small_dataset(data_path, n, output_path, seed=42):
    """
    Create a small dataset of size n from the original dataset.

    Args:
    data_path (str): Path to the original dataset
    n (int): Number of samples to include in the small dataset
    output_path (str): Path to save the small dataset
    seed (int): Random seed for reproducibility
    """
    random.seed(seed)

    # Load the original data
    states = np.load(f"{data_path}/states.npy", mmap_mode="r")
    actions = np.load(f"{data_path}/actions.npy")

    # Check if locations exist
    try:
        locations = np.load(f"{data_path}/locations.npy")
        has_locations = True
    except FileNotFoundError:
        has_locations = False

    # Get total number of samples
    total_samples = len(states)

    # Randomly select n indices
    selected_indices = random.sample(range(total_samples), n)

    # Create small datasets
    small_states = states[selected_indices]
    small_actions = actions[selected_indices]

    # Save small datasets
    np.save(f"{output_path}/states.npy", small_states)
    np.save(f"{output_path}/actions.npy", small_actions)

    if has_locations:
        small_locations = locations[selected_indices]
        np.save(f"{output_path}/locations.npy", small_locations)

    print(f"Created small dataset with {n} samples at {output_path}")


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    p_augment_data=0, # probablity of augmenting
    train=True,
    p_noise=None,
    p_flip=None,
    p_rotate=None,
    noise_std=0.05
):
    if not train and p_augment_data > 0.0:
        raise ValueError("Don't augment probe data pls")
    assert 0 <= p_augment_data <= 1
    print(f"Loading data from {data_path} ...")
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        p_aug=p_augment_data,
        p_flip=p_flip,
        p_rotate=p_rotate,
        p_noise=p_noise,
        noise_std=noise_std,
        cache_size=batch_size*32
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
        num_workers=4,    # Use multiple workers for data loading
        persistent_workers=True  # Keep worker processes alive between epochs
    )

    return loader
