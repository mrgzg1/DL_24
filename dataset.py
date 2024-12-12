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
    aug_image = image.copy()
    aug_action = action.copy()
    aug_image[:, 0, :, :] = np.flip(aug_image[:, 0, :, :], axis=2)  # Flip agent horizontally
    aug_image[:, 1, :, :] = np.flip(aug_image[:, 1, :, :], axis=2)  # Flip walls and borders horizontally
    aug_action[:, 0] = -aug_action[:, 0]  # Reverse x-component of actions
    return aug_image, aug_action

def vertical_flip(image, action):
    """Apply vertical flip augmentation to image and adjust action accordingly"""
    aug_image = image.copy()
    aug_action = action.copy()
    aug_image[:, 0, :, :] = np.flip(aug_image[:, 0, :, :], axis=1)  # Flip agent vertically
    aug_image[:, 1, :, :] = np.flip(aug_image[:, 1, :, :], axis=1)  # Flip walls and borders vertically
    aug_action[:, 1] = -aug_action[:, 1]  # Reverse y-component of actions
    return aug_image, aug_action

def rotate_90(image, action):
    """
    Rotate image by 90 degrees clockwise and adjust action vector accordingly
    
    Args:
        image: numpy array of shape (B, C, H, W)
        action: numpy array of shape (B, 2) containing (dx, dy) vectors
    """
    aug_image = image.copy()
    aug_action = action.copy()
    
    # Rotate both channels (agent and walls) 90 degrees clockwise
    aug_image[:, 0, :, :] = np.rot90(aug_image[:, 0, :, :], k=1, axes=(1,2))
    aug_image[:, 1, :, :] = np.rot90(aug_image[:, 1, :, :], k=1, axes=(1,2))
    
    # For 90 degree clockwise rotation:
    # x' = y
    # y' = -x
    old_x = aug_action[:, 0].copy()
    aug_action[:, 0] = aug_action[:, 1]
    aug_action[:, 1] = -old_x
    
    return aug_image, aug_action

def add_gaussian_noise(image, std=0.05):
    """Add Gaussian noise to image"""
    aug_image = image.copy()
    noise = np.random.normal(0, std, aug_image.shape)
    aug_image += noise
    aug_image = np.clip(aug_image, 0, 1)  # Ensure values remain in valid range [0, 1]
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
    aug_image = image.copy()
    aug_action = action.copy()
    
    # Set individual probabilities to overall p_aug if not specified
    p_hflip = p_hflip if p_hflip is not None else p_aug
    p_vflip = p_vflip if p_vflip is not None else p_aug
    p_rot90 = p_rot90 if p_rot90 is not None else p_aug
    p_noise = p_noise if p_noise is not None else p_aug
    
    # Only apply augmentations with probability p_aug
    if np.random.rand() < p_aug:
        # Horizontal flip
        if np.random.rand() < p_hflip:
            aug_image, aug_action = horizontal_flip(aug_image, aug_action)
            
        # Vertical flip    
        if np.random.rand() < p_vflip:
            aug_image, aug_action = vertical_flip(aug_image, aug_action)
            
        # 90 degree rotation
        if np.random.rand() < p_rot90:
            aug_image, aug_action = rotate_90(aug_image, aug_action)
            
        # Add noise
        if np.random.rand() < p_noise:
            aug_image = add_gaussian_noise(aug_image, noise_std)
    
    return aug_image, aug_action


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        apply_augs=False,
        p_flip=0.5
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")
        self.p_flip = p_flip

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        print(f"Dataset size: {len(self.states)}")
        print(f"States shape: {self.states.shape}")
        print(f"Actions shape: {self.actions.shape}")

        if apply_augs:
            # Augment the dataset
            aug_states, aug_actions = self.augment_dataset(self.states, self.actions)
            self.states = np.concatenate([self.states, aug_states], axis=0)
            self.actions = np.concatenate([self.actions, aug_actions], axis=0)

            print(f"Augmented dataset size: {len(self.states)}")
            print(f"Augmented states shape: {self.states.shape}")
            print(f"Augmented actions shape: {self.actions.shape}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
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

    def augment_dataset(self, images, actions):
        """
        Augments the dataset with the defined augmentations and adjusts actions.

        Args:
            images (numpy.ndarray): Original images of shape (N, T, 2, 65, 65).
            actions (numpy.ndarray): Original actions of shape (N, T-1, 2).

        Returns:
            aug_images (numpy.ndarray): Augmented images of shape (2 * N, T, 2, 65, 65).
            aug_actions (numpy.ndarray): Adjusted actions of shape (2 * N, T-1, 2).
        """
        N, T, _, H, W = images.shape
        aug_images = []
        aug_actions = []

        for i in range(N):
            # Add original data
            aug_images.append(images[i])
            aug_actions.append(actions[i])

            # Apply augmentations
            aug_image, aug_action = apply_augmentations(images[i], actions[i], p_flip=self.p_flip)
            
            # Append augmented data
            aug_images.append(aug_image)
            aug_actions.append(aug_action)

        # Convert to numpy arrays
        aug_images = np.array(aug_images)
        aug_actions = np.array(aug_actions)

        return aug_images, aug_actions

    # Display a trajectory of observations using matplotlib
    def display_trajectory(self, i):
        states = self.states[i]
        actions = self.actions[i]

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
    train=True,
):
    print(f"Loading data from {data_path} ...")
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
