import numpy as np
import wandb

# This implementation uses PyTorch. Please ensure you have it installed.
# pip install torch torchvision

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


class DemonstrationDataset(Dataset):
    """A PyTorch Dataset for handling RLBench demonstration data."""

    def __init__(self, demos, task_variations):
        self.samples = []
        demos_per_variation = len(demos) // task_variations

        for i, demo in enumerate(demos):
            # Determine the variation index for this demo
            variation_index = i // demos_per_variation
            goal_vector = np.zeros(task_variations, dtype=np.float32)
            goal_vector[variation_index] = 1.0

            for obs in demo:
                # For each step in the demo, we store the observation, goal, and action
                self.samples.append((obs.wrist_rgb, goal_vector, obs.joint_velocities))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, goal, action = self.samples[idx]
        # Convert to tensors
        img_tensor = torch.from_numpy(img).float()
        goal_tensor = torch.from_numpy(goal).float()
        action_tensor = torch.from_numpy(action).float()
        return img_tensor, goal_tensor, action_tensor


class BCPolicy(nn.Module):
    """A goal-conditioned policy that uses a pre-trained ResNet18 backbone."""

    def __init__(self, obs_shape, goal_shape, action_shape):
        super(BCPolicy, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        vision_feature_size = 512
        self.policy_head = nn.Sequential(
            nn.Linear(vision_feature_size + goal_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape),
            nn.Tanh()
        )

    def forward(self, image_obs, goal_vec):
        image_obs = image_obs.permute(0, 3, 1, 2)
        vision_features = self.vision_encoder(image_obs)
        vision_features = vision_features.view(vision_features.size(0), -1)
        combined_features = torch.cat([vision_features, goal_vec], dim=1)
        action = self.policy_head(combined_features)
        return action


class BCAgent(object):
    """A Behavioral Cloning agent that manages the policy and training."""

    def __init__(self, action_shape, obs_shape, goal_shape):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.policy = BCPolicy(
            obs_shape, goal_shape, action_shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def ingest_demos(self, demos, task_variations):
        print(f"Starting BC training on {len(demos)} demonstrations...")
        
        dataset = DemonstrationDataset(demos, task_variations)
        # Set batch_size to a value like 32 or 64
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        num_epochs = 10 # Train for 10 epochs as an example
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_imgs, batch_goals, batch_expert_actions in dataloader:
                batch_imgs = batch_imgs.to(self.device)
                batch_goals = batch_goals.to(self.device)
                batch_expert_actions = batch_expert_actions.to(self.device)
                
                self.policy.train() # Set policy to training mode
                predicted_actions = self.policy(batch_imgs, batch_goals)
                loss = self.loss_fn(predicted_actions, batch_expert_actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
            # Log the training loss to wandb
            wandb.log({"training_loss": avg_loss, "epoch": epoch + 1})

        print("BC Training complete.")
        torch.save(self.policy.state_dict(), 'bc_policy.pth')
        print("Saved BC policy weights to bc_policy.pth")

    def act(self, obs, goal_vector):
        self.policy.eval() # Set the policy to evaluation mode
        img = torch.from_numpy(obs.wrist_rgb).float().unsqueeze(0).to(self.device)
        goal = torch.from_numpy(goal_vector).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy(img, goal).squeeze(0).cpu().numpy()
        
        return action