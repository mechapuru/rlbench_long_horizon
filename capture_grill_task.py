
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.long_horizon_grill_task import LongHorizonGrillTask
import matplotlib.pyplot as plt
import os

# Create a directory to save the images
output_dir = "grill_task_capture"
os.makedirs(output_dir, exist_ok=True)

# Setup observation configuration to get RGB and depth images
obs_config = ObservationConfig()
obs_config.front_camera.rgb = True
obs_config.front_camera.depth = True
obs_config.left_shoulder_camera.rgb = True
obs_config.left_shoulder_camera.depth = True
obs_config.right_shoulder_camera.rgb = True
obs_config.right_shoulder_camera.depth = True
obs_config.overhead_camera.rgb = True
obs_config.overhead_camera.depth = True
obs_config.wrist_camera.rgb = True
obs_config.wrist_camera.depth = True

# Set image size
obs_config.front_camera.image_size = (512, 512)
obs_config.left_shoulder_camera.image_size = (512, 512)
obs_config.right_shoulder_camera.image_size = (512, 512)
obs_config.overhead_camera.image_size = (512, 512)
obs_config.wrist_camera.image_size = (512, 512)

# Initialize the environment
env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointPosition(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False,
)
env.launch()

# Get the task
task = env.get_task(LongHorizonGrillTask)
descriptions, obs = task.reset()

print("Saving images and camera parameters...")

# Save images
camera_types = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
for cam_type in camera_types:
    # Save RGB image
    rgb_image = getattr(obs, f"{cam_type}_rgb")
    plt.imsave(os.path.join(output_dir, f"{cam_type}_rgb.png"), rgb_image)

    # Save depth image
    depth_image = getattr(obs, f"{cam_type}_depth")
    np.save(os.path.join(output_dir, f"{cam_type}_depth.npy"), depth_image)
    # Save a visualized depth image
    plt.imsave(os.path.join(output_dir, f"{cam_type}_depth.png"), depth_image, cmap='viridis')

print(f"Images saved to '{output_dir}' directory.")

# Print camera parameters
for cam_type in camera_types:
    camera = getattr(env.scene, f"{cam_type}_camera")
    print(f"--- {cam_type} Camera ---")
    print("Intrinsics:", camera.get_intrinsic_matrix())
    print("Extrinsics:", camera.get_matrix())

print("Simulation finished. Shutting down environment.")
env.shutdown()
