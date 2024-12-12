from typing import NamedTuple, Optional
import torch
import numpy as np
from matplotlib import pyplot as plt


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        print(f"Dataset size: {len(self.states)}")
        print(f"States shape: {self.states.shape}")
        print(f"Actions shape: {self.actions.shape}")


        # Augment the dataset
        aug_states, aug_actions = self.augment_dataset(self.states, self.actions)

        self.states = np.concatenate([self.states, aug_states], axis=0)
        self.actions = np.concatenate([self.actions, aug_actions], axis=0)

        print(f"Dataset size: {len(self.states)}")
        print(f"States shape: {self.states.shape}")
        print(f"Actions shape: {self.actions.shape}")



        

        # self._print_data_stats()
        
        # Display a 5 random trajectories
        # for i in range(5):
        #     self.display_trajectory(i)


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

            # Create augmented image
            aug_image = images[i].copy()
            aug_action = actions[i].copy()

            # Horizontal flip
            if np.random.rand() < 0.5:
                aug_image[:, 0, :, :] = np.flip(aug_image[:, 0, :, :], axis=2)  # Flip agent horizontally
                aug_image[:, 1, :, :] = np.flip(aug_image[:, 1, :, :], axis=2)  # Flip walls and borders horizontally
                aug_action[:, 0] = -aug_action[:, 0]  # Reverse x-component of actions

            # Vertical flip
            if np.random.rand() < 0.5:
                aug_image[:, 0, :, :] = np.flip(aug_image[:, 0, :, :], axis=1)  # Flip agent vertically
                aug_image[:, 1, :, :] = np.flip(aug_image[:, 1, :, :], axis=1)  # Flip walls and borders vertically
                aug_action[:, 1] = -aug_action[:, 1]  # Reverse y-component of actions

            # Add Gaussian noise
            noise = np.random.normal(0, 0.05, aug_image.shape)
            aug_image += noise
            aug_image = np.clip(aug_image, 0, 1)  # Ensure values remain in valid range [0, 1]

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
