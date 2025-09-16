import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PickAndLift, PutItemInDrawer


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


# Define the sequence of tasks for our "super task"
super_task_sequence = [ReachTarget, PickAndLift, PutItemInDrawer]

obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False)
env.launch()

agent = Agent(env.action_shape)

for task_class in super_task_sequence:
    print("Starting task: %s" % task_class.__name__)
    task = env.get_task(task_class)
    descriptions, obs = task.reset()
    print(descriptions)
    
    done = False
    while not done:
        action = agent.act(obs)
        try:
            obs, reward, done = task.step(action)
            success, _ = task.success()
            if success:
                print(f"Task {task_class.__name__} successful!")
                done = True
        except Exception as e:
            print(f"An error occurred during task {task_class.__name__}: {e}")
            done = True


print("Super task complete!")
env.shutdown()
