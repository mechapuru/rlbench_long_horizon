import numpy as np
from PIL import Image
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import OpenGrill


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=True)
env.launch()

task = env.get_task(OpenGrill)

agent = Agent(env.action_shape)

training_steps = 120
episode_length = 40
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
        
        # Step the environment once to allow the renderer to capture the scene
        obs, _, _ = task.step(agent.act(obs))
        
        # Save the image
        # img = Image.fromarray(obs.front_rgb)
        # img.save("front_camera_capture.png")
        # print("Saved front_camera_capture.png")
    action = agent.act(obs)
    print(action)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
