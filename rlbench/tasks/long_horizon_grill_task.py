from typing import List
import logging
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, JointCondition

class LongHorizonGrillTask(Task):

    def init_task(self) -> None:
        # Configure logging to write to a file
        logging.basicConfig(level=logging.INFO, 
                            filename="output.txt", 
                            filemode="w",
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('LongHorizonGrillTask: init_task called.')

        # Get handles to objects needed for this sub-task
        self.steak = Shape('steak')
        self.chicken = Shape('chicken')
        self.plate = Shape('plate')
        self.lid_joint = Joint('lid_joint')

        self.success_sensor_grill = ProximitySensor('success_meat_on_grill')
        self.success_sensor_plate_source = ProximitySensor('success_source')
        self.success_sensor_plate_target = ProximitySensor('success_target')
        self.success_sensor_meat_plate = ProximitySensor('success_meat_on_plate')

        self.register_graspable_objects([self.steak, self.chicken, self.plate])

        self.register_success_conditions([
            DetectedCondition(self.chicken, self.success_sensor_grill),
            DetectedCondition(self.plate, self.success_sensor_plate_source, negated=True),
            DetectedCondition(self.plate, self.success_sensor_plate_target),
            DetectedCondition(self.chicken, self.success_sensor_meat_plate),
            JointCondition(self.lid_joint, np.deg2rad(30)),
            NothingGrasped(self.robot.gripper)
        ])
        logging.info('Success conditions registered.')


    def init_episode(self, index: int) -> List[str]:
        # logging.info(f'LongHorizonGrillTask: init_episode called for variation {index}.')

        return ['put the meat on the grill, take the plate off the rack, and close the grill']

    def variation_count(self) -> int:
        return 1