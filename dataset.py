from typing import NamedTuple
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

def horizontal_flip(image, action):
    aug_image = torch.flip(image, dims=[3])  # Flip horizontally
    aug_action = action.clone()
    aug_action[:, 0] = -aug_action[:, 0]
    return aug_image, aug_action

def vertical_flip(image, action):
    aug_image = torch.flip(image, dims=[2])  # Flip vertically
    aug_action = action.clone()
    aug_action[:, 1] = -aug_action[:, 1]
    return aug_image, aug_action

def rotate_90(image, action):
    aug_image = torch.rot90(image, k=1, dims=[2, 3])
    aug_action = action.clone()
    old_x = aug_action[:, 0].clone()
    aug_action[:, 0] = aug_action[:, 1]
    aug_action[:, 1] = -old_x
    return aug_image, aug_action

def add_gaussian_noise(image, std=0.05):
    aug_image = image.clone()
    noise = torch.randn_like(aug_image[:, 1]) * std
    aug_image[:, 1] = torch.clamp(aug_image[:, 1] + noise, 0, 1)
    return aug_image

def apply_augmentations(image, action, p_aug=0.5, p_hflip=None, p_vflip=None, p_rot90=None, p_noise=None, noise_std=0.05):
    p_hflip = p_hflip if p_hflip is not None else p_aug
    p_vflip = p_vflip if p_vflip is not None else p_aug
    p_rot90 = p_rot90 if p_rot90 is not None else p_aug
    p_noise = p_noise if p_noise is not None else p_aug

    if torch.rand(1).item() < p_aug:
        if torch.rand(1).item() < p_hflip:
            image, action = horizontal_flip(image, action)
        if torch.rand(1).item() < p_vflip:
            image, action = vertical_flip(image, action)
        if torch.rand(1).item() < p_rot90:
            image, action = rotate_90(image, action)
        if torch.rand(1).item() < p_noise:
            image = add_gaussian_noise(image, std=noise_std)
    return image, action

class WallDataset(Dataset):
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
        cache_size=128,  # Prefetch this many samples into GPU
    ):
        self.device = device
        self.cache_size = cache_size
        self.data_cache = {
            "states": None,
            "actions": None,
            "locations": None
        }
        self.cache_start = 0

        # Memory-map the data to avoid loading it all into CPU memory at once
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy", mmap_mode="r")

        self.locations = None
        if probing:
            try:
                self.locations = np.load(f"{data_path}/locations.npy", mmap_mode="r")
            except FileNotFoundError:
                print("Warning: locations.npy not found, skipping location loading.")

        self.apply_augs = (p_aug > 0)
        self.p_aug = p_aug
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise
        self.noise_std = noise_std

        # Initialize cache with first batch
        self._load_to_cache(0)
        self._print_data_stats()

    def __len__(self):
        return len(self.states)

    def _load_to_cache(self, start_idx):
        end_idx = min(start_idx + self.cache_size, len(self.states))

        # Convert the required slice from numpy to torch tensors
        states = torch.from_numpy(self.states[start_idx:end_idx]).float()
        actions = torch.from_numpy(self.actions[start_idx:end_idx]).float()

        locations = None
        if self.locations is not None:
            locations = torch.from_numpy(self.locations[start_idx:end_idx]).float()

        # Move to GPU
        self.data_cache["states"] = states.to(self.device, non_blocking=True)
        self.data_cache["actions"] = actions.to(self.device, non_blocking=True)
        self.data_cache["locations"] = locations.to(self.device, non_blocking=True) if locations is not None else None
        
        self.cache_start = start_idx

    def __getitem__(self, i):
        if not (self.cache_start <= i < self.cache_start + self.cache_size):
            self._load_to_cache(i)

        cache_idx = i - self.cache_start
        states = self.data_cache["states"][cache_idx]
        actions = self.data_cache["actions"][cache_idx]
        locations = self.data_cache["locations"][cache_idx] if self.data_cache["locations"] is not None else torch.empty(0, device=self.device)

        # Apply augmentations if needed (on GPU)
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
        # Note: This will load one sample fully into CPU memory.
        # For large datasets this is okay as it's just for visualization.
        states = torch.from_numpy(self.states[i]).cpu().numpy()
        actions = torch.from_numpy(self.actions[i]).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title("Trajectory of the agent")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        for t in range(states.shape[0]):
            state = states[t]
            wall_image = state[0]
            agent_image = state[1]
            ax.imshow(wall_image, cmap="gray")
            ax.imshow(agent_image, cmap="jet", alpha=0.5)
            plt.pause(0.5)
            plt.draw()

        plt.show()
        plt.close()

def create_small_dataset(data_path, n, output_path, seed=42):
    random.seed(seed)
    states = np.load(f"{data_path}/states.npy", mmap_mode="r")
    actions = np.load(f"{data_path}/actions.npy")

    try:
        locations = np.load(f"{data_path}/locations.npy", mmap_mode="r")
        has_locations = True
    except FileNotFoundError:
        has_locations = False

    total_samples = len(states)
    selected_indices = random.sample(range(total_samples), n)

    small_states = states[selected_indices]
    small_actions = actions[selected_indices]

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
    p_augment_data=0,
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

    # Adjust cache_size as desired, for example batch_size*64
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        p_aug=p_augment_data,
        p_flip=p_flip,
        p_rotate=p_rotate,
        p_noise=p_noise,
        noise_std=noise_std,
        cache_size=batch_size * 64
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    return loader
