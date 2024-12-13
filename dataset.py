from typing import NamedTuple, Optional
import random
import torch
import numpy as np
from matplotlib import pyplot as plt


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
    """Add Gaussian noise to image"""
    noise = torch.randn_like(image) * std
    aug_image = image + noise
    aug_image = torch.clamp(aug_image, 0, 1)  # Ensure values remain in valid range [0, 1]
    return aug_image

def apply_augmentations(image, action, p_aug=0.5, p_hflip=None, p_vflip=None, p_rot90=None, p_noise=None, noise_std=0.05):
    """Apply all augmentations with given probabilities
    
    Args:
        image: Input image to augment
        action: Input action to adjust
        p_aug: Overall probability of applying any augmentation
        p_hflip: Probability of horizontal flip (if None, uses p_aug)
        p_vflip: Probability of vertical flip (if None, uses p_aug)
        p_rot90: Probability of 90 degree rotation (if None, uses p_aug)
        p_noise: Probability of adding noise (if None, uses p_aug)
        noise_std: Standard deviation for Gaussian noise
    """
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
            aug_image = add_gaussian_noise(aug_image, noise_std)
    
    return aug_image, aug_action


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        apply_augs=False,
        p_aug=0.1
    ):
        self.device = device
        self.states = torch.from_numpy(np.load(f"{data_path}/states.npy", mmap_mode="r")).float().to(device)
        self.actions = torch.from_numpy(np.load(f"{data_path}/actions.npy")).float().to(device)
        self.apply_augs = apply_augs
        self.p_aug = p_aug

        if probing:
            self.locations = torch.from_numpy(np.load(f"{data_path}/locations.npy")).float().to(device)
        else:
            self.locations = None

        print(f"Dataset size: {len(self.states)}")
        print(f"States shape: {self.states.shape}")
        print(f"Actions shape: {self.actions.shape}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = self.states[i]
        actions = self.actions[i]

        if self.apply_augs:
            states, actions = apply_augmentations(states, actions, p_aug=self.p_aug)

        if self.locations is not None:
            locations = self.locations[i]
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)

    def _print_data_stats(self):
        print("Data statistics:")
        print(f"States: {self.states.shape}")
        print(f"Actions: {self.actions.shape}")

        # Print max and min pixel values
        print(f"States max: {self.states.max()}")

        # Print variance of pixel values
        print(f"States variance: {self.states.var()}")

        # Print max and min action values
        print(f"Actions max: {self.actions.max()}", f"Actions min: {self.actions.min()}")
        print()

    # Display a trajectory of observations using matplotlib
    def display_trajectory(self, i):
        states = self.states[i].cpu().numpy()
        actions = self.actions[i].cpu().numpy()

        # Image has 2 channel dimension. Overlay the two images to get the final image
        # The first channel is the wall image and the second channel is the agent image

        # Display the trajectory of the agent
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title("Trajectory of the agent")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        for t in range(states.shape[0]):
            state = states[t]
    
            # Display the wall image
            wall_image = state[0]
            # Print max and min pixel values and average pixel value
            print(f"Wall image max: {wall_image.max()}", f"Wall image min: {wall_image.min()}", f"Wall image mean: {wall_image.mean()}")
            ax.imshow(wall_image, cmap="gray")

            # Display the agent image
            agent_image = state[1]
            # Print max and min pixel values
            # print(f"Agent image max: {agent_image.max()}", f"Agent image min: {agent_image.min()}", f"Agent image mean: {agent_image.mean()}")
            ax.imshow(agent_image, cmap="jet", alpha=0.5)
            
            plt.pause(0.5)
            plt.draw()
            
        plt.show()

        # Kill the plot
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
):
    if not train and p_augment_data > 0.0:
        raise ValueError("Don't augment probe data pls")
    assert 0 <= p_augment_data <= 1
    print(f"Loading data from {data_path} ...")
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        apply_augs=True if p_augment_data > 0.0 else False,
        p_aug=p_augment_data
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
