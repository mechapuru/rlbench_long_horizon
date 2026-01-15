import numpy as np
import wandb

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import MeatOnGrill

# Import the agent from our new file
from bc_agent import BCAgent

def main():
    """Main training script."""

    # To use wandb, you need to install it (pip install wandb) and login (wandb login)
    wandb.init(project="rlbench-bc-training", entity="paddyd")

    # --- 1. ENVIRONMENT SETUP ---
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.wrist_rgb.enabled = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

    env = Environment(
        action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(MeatOnGrill)
    task_variations = task.variation_count()

    # --- 2. DATA GENERATION & BC TRAINING ---
    print(f"Generating demonstrations for {task_variations} variations...")
    demos = []
    demos_per_variation = 5
    for i in range(task_variations):
        variation_demos = task.get_demos(
            demos_per_variation, live_demos=True, variation_number=i)
        demos.extend(variation_demos)
    print(f"Demonstration generation complete. Total demos: {len(demos)}.")

    # Initialize the agent
    agent = BCAgent(
        action_shape=env.action_shape[0], 
        obs_shape=(3, 128, 128), # (channels, height, width)
        goal_shape=task_variations)

    # Train the agent using the demonstrations.
    agent.ingest_demos(demos, task_variations)


    # --- 3. EVALUATION ---
    print("\n--- Evaluating the BC-trained agent ---")
    episodes = 4
    for i in range(episodes):
        variation_index = i % task_variations
        task.set_variation(variation_index)
        
        print(f'--- Starting Evaluation Episode {i+1}/{episodes} (Variation: {variation_index}) ---')
        descriptions, obs = task.reset()
        print("Task Description: ", descriptions[0])

        goal_vector = np.zeros(task_variations)
        goal_vector[variation_index] = 1.0

        episode_success = False
        for step in range(100): # Max 100 steps per episode
            action = agent.act(obs, goal_vector)
            try:
                obs, reward, terminate = task.step(action)
                if terminate:
                    print("Episode terminated successfully!")
                    episode_success = True
                    break
            except Exception as e:
                print(f"An error occurred during task step: {e}")
                break
        else:
            print("Episode reached max steps.")
        
        # Log the success of the episode to wandb
        wandb.log({
            f'eval_success_var_{variation_index}': 1.0 if episode_success else 0.0,
            'episode_number': i
        })

    print('Done')
    env.shutdown()

if __name__ == "__main__":
    main()