"""
Script to inspect the state of ANY task in RLBench.

Usage:
    python examples/inspect_task_state.py --task OpenGrill

TROUBLESHOOTING:
If you encounter a crash with "version `GLIBCXX_3.4.30' not found", it is a conflict
between Conda's libstdc++ and your system's graphics drivers.

FIX:
Preload your system's libstdc++ before running the script:
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
"""

import numpy as np
import os
import argparse
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg') # Prevent Qt conflicts
import matplotlib.pyplot as plt
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from pyrep.const import ObjectType
import rlbench.tasks as tasks

def get_task_class(task_name):
    # Try to get the task class from the rlbench.tasks module
    try:
        # Check if the task name is a class name in rlbench.tasks
        if hasattr(tasks, task_name):
            return getattr(tasks, task_name)
        
        # If not found directly, try case-insensitive search or snake_case conversion
        # But usually users provide the CamelCase class name.
        # Let's recursively search if needed, but the main init has most.
        raise AttributeError(f"Task '{task_name}' not found in rlbench.tasks")
    except ImportError as e:
        print(f"Error importing task: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Inspect RLBench Task State')
    parser.add_argument('--task', type=str, required=True, help='Name of the task class (e.g., OpenGrill)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (default: True)')
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    task_name = args.task
    save_dir = f"{task_name}_visuals"
    
    # 1. Get the Task Class
    try:
        TaskClass = get_task_class(task_name)
    except Exception as e:
        print(f"Failed to find task class '{task_name}'. Make sure it is a valid RLBench task.")
        print(e)
        return

    # 2. Create Observation Config and enable EVERYTHING
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    
    # 3. Define Action Mode
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), 
        gripper_action_mode=Discrete()
    )

    # 4. Initialize Environment
    env = Environment(
        action_mode, 
        obs_config=obs_config, 
        headless=args.headless
    )
    
    try:
        env.launch()
        
        # 5. Get the Task
        print(f"Loading task: {task_name}...")
        task = env.get_task(TaskClass)
        
        # 6. Reset to get the initial observation
        print("Resetting task to get specific state information...")
        descriptions, obs = task.reset()
        
        print("\n" + "="*80)
        print(f"Task: {task.get_name()}")
        print(f"Variation Descriptions: {descriptions}")
        print(f"Output Directory: {os.path.abspath(save_dir)}")
        print("="*80 + "\n")

        # 7. Inspect the Observation Object
        print("State Information Available (Observation Object):")
        
        # Helper to print info
        def print_info(name, value):
            if value is None:
                print(f"  {name:<30}: None (Not Enabled)")
            elif isinstance(value, np.ndarray):
                print(f"  {name:<30}: Shape {value.shape}, Dtype {value.dtype}, Range [{value.min():.2f}, {value.max():.2f}]")
            elif isinstance(value, list) or isinstance(value, float) or isinstance(value, int):
                print(f"  {name:<30}: {value}")
            else:
                print(f"  {name:<30}: Type {type(value)}")

        # Cameras
        cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']
        camera_types = ['rgb', 'depth', 'mask', 'point_cloud']
        
        print(f"\n--- Visual State ({len(cameras)} cameras) ---")
        for cam in cameras:
            for ctype in camera_types:
                attr = f"{cam}_{ctype}"
                val = getattr(obs, attr, None)
                print_info(attr, val)

        # Robot State
        print("\n--- Robot Proprioceptive State ---")
        robot_attrs = [
            'joint_positions', 'joint_velocities', 'joint_forces',
            'gripper_pose', 'gripper_open', 'gripper_matrix', 
            'gripper_joint_positions', 'gripper_touch_forces'
        ]
        for attr in robot_attrs:
            val = getattr(obs, attr, None)
            print_info(attr, val)

        # Task State
        print("\n--- Task Specific State ---")
        print_info('task_low_dim_state', obs.task_low_dim_state)
        
        # Misc
        print("\n--- Miscellaneous / Metadata ---")
        if obs.misc:
            for k, v in obs.misc.items():
                print_info(f"misc['{k}']", v)
        else:
            print("  misc: Empty")

        # --- Breakdown of task_low_dim_state ---
        print("\n" + "="*80)
        print(f"Breaking down 'task_low_dim_state' for {task_name}:")
        print("="*80)
        
        # Prepare text output buffer
        output_lines = []
        
        # Access internal task object
        if hasattr(task, '_task') and hasattr(task._task, '_initial_objs_in_scene'):
            total_dims = 0
            
            output_lines.append(f"{'Object Name':<40} | {'Type':<15} | {'Dims':<5} | {'Description'}")
            output_lines.append("-" * 100)
            
            # Print headers to console
            print(output_lines[0])
            print(output_lines[1])
            
            for obj, obj_type in task._task._initial_objs_in_scene:
                dims = 0
                desc = ""
                
                if not obj.still_exists():
                    desc = "Deleted (Zero-filled)"
                    dims = 7
                    if obj_type == ObjectType.JOINT: dims += 1
                    elif obj_type == ObjectType.FORCE_SENSOR: dims += 6
                else:
                    dims = 7 # Pose (3 pos + 4 quat)
                    desc = "Pose (7)"
                    if obj_type == ObjectType.JOINT:
                        dims += 1
                        desc += " + Joint Position (1)"
                    elif obj_type == ObjectType.FORCE_SENSOR:
                        dims += 6
                        desc += " + Force/Torque (6)"
                
                line = f"{obj.get_name():<40} | {str(obj_type).split('.')[-1]:<15} | {dims:<5} | {desc}"
                print(line)
                output_lines.append(line)
                total_dims += dims
            
            output_lines.append("-" * 100)
            output_lines.append(f"Total Calculated Dimensions: {total_dims}")
            output_lines.append(f"Actual Observation Shape:    {obs.task_low_dim_state.shape[0]}")
        else:
            msg = "Could not access _initial_objs_in_scene (via _task) to decompose state. Saving raw values only."
            print(msg)
            output_lines.append(msg)

        output_lines.append("\n=== Raw Values ===")
        output_lines.append(str(list(obs.task_low_dim_state)))
        
        # Save to file (ALWAYS)
        os.makedirs(save_dir, exist_ok=True)
        txt_path = os.path.join(save_dir, "task_low_dim_state.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"Saved low-dim state data to: {os.path.abspath(txt_path)}")

        # --- Saving Images and Masks ---
        print("\n" + "="*80)
        print("Saving Visual Observations...")
        print("="*80)
        
        print(f"Saving images to: {os.path.abspath(save_dir)}")
        
        for cam in cameras:
            # 1. Save RGB
            rgb_attr = f"{cam}_rgb"
            rgb_data = getattr(obs, rgb_attr, None)
            if rgb_data is not None:
                img_path = os.path.join(save_dir, f"{cam}_rgb.png")
                Image.fromarray(rgb_data).save(img_path)
                print(f"  Saved {img_path}")
                
            # 2. Save Mask
            mask_attr = f"{cam}_mask"
            mask_data = getattr(obs, mask_attr, None)
            if mask_data is not None:
                mask_path = os.path.join(save_dir, f"{cam}_mask.png")
                
                # Remap unique handles to indices for better visualization
                unique_handles = np.unique(mask_data)
                remapped_mask = np.zeros_like(mask_data)
                for i, h in enumerate(unique_handles):
                    remapped_mask[mask_data == h] = i
                
                # Using nipy_spectral for high contrast between indices
                plt.imsave(mask_path, remapped_mask, cmap='nipy_spectral')
                print(f"  Saved {mask_path} (Colormapped)")

        print("\nDone.")

    finally:
        env.shutdown()

if __name__ == "__main__":
    main()
