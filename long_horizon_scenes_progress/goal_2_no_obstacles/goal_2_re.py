from typing import List, Dict, cast, Optional
import numpy as np

from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor

from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedCondition,
    DetectedSeveralCondition,
    Condition,
)
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.exceptions import BoundaryError


# ---- waypoint indices ----
SHOES_LAST_WP_IDX = 12

WP_STAGE_FIXED = 13  # air staging over groceries boundary centre (validator-safe)

WP_PICK_APPROACH = 14  # approach above *_grasp_point
WP_PICK_GRASP = 15     # exact *_grasp_point
WP_PICK_LIFT = 16      # lift from grasp
WP_TRANSFER_UP = 17    # straight-up raise from WP16 to shelf height (world +Z)
WP_PLACE_APPROACH = 18 # == goal_<item>
WP_PLACE_PLACE = 19    # scene-defined place pose
WP_PLACE_RETREAT = 20  # retreat back to 18

# ---- deltas (meters) ----
PICK_APPROACH_DZ = 0.20
PICK_LIFT_DZ = 0.22

# 16 -> 17 safety rails (vertical raise robustness)
TRANSFER_MIN_RAISE = 0.28
TRANSFER_MAX_RAISE = 0.40

# Stage (WP13) safety
STAGE_Z_ABOVE_TABLE = 0.42
STAGE_Z_ABOVE_GRASP = 0.28
STAGE_CLEAR_DZ = 0.16

# Limit the twist (roll) change around the goal's local Z during 17->18
TWIST_LIMIT_17_TO_18_DEG = 70.0
TWIST_LIMIT_17_TO_18_RAD = np.deg2rad(TWIST_LIMIT_17_TO_18_DEG)

# Small push to insert WP19 a bit deeper (+X in world)
WP19_INSERT_WORLD_DX = 0.03

# Order = pick loop order
GROCERY_NAMES = ["soup", "spam"]


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply (x,y,z,w) order."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=float)


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    s = np.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)], dtype=float)


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


class Goal2Re(Task):
    """Pick (14–16), raise (17), approach (18 = goal), place (19), retreat (20).

    Gripper policy:
      • OPEN on WP14 and while moving to WP15.
      • GRASP + CLOSE at the start of WP16 (i.e., after arriving to WP15).
      • From WP16→WP18: re-CLOSE + re-GRASP at each waypoint start (defeats any injected opens).
      • Hard-lock object to tool tip until WP19 (parenting), then unlock.
      • RELEASE only at WP19. WP19 pose is scene-defined (with small +X insert).
    """

    # -------------------- SHOES PHASE --------------------
    def _init_shoe_phase(self) -> None:
        self.shoe1 = Shape("shoe1")
        self.shoe2 = Shape("shoe2")
        self.box_lid = Shape("box_lid")
        self.box_joint = Joint("box_joint")
        self.success_sensor_shoe = ProximitySensor("success_in_box")

        self._pre_grocery_wp: Dict[int, Dummy] = {}
        for i in range(WP_STAGE_FIXED):  # 0..12
            try:
                self._pre_grocery_wp[i] = Dummy(f"waypoint{i}")
            except Exception:
                pass
        for i in self._pre_grocery_wp.keys():
            self.register_waypoint_ability_start(i, self._noop_shoes_when_done)

    # -------------------- GROCERIES PHASE --------------------
    def _init_grocery_phase(self) -> None:
        self.groceries: List[Shape] = [Shape(n.replace(" ", "_")) for n in GROCERY_NAMES]
        self.success_sensor_grocery = ProximitySensor("success")
        self.groceries_to_place = len(self.groceries)
        self.groceries_placed = 0

        # track held object + whether it's hard-locked
        self._held_obj: Optional[Shape] = None
        self._hard_locked: bool = False

        self.grasp_points = [Dummy(f"{n.replace(' ', '_')}_grasp_point") for n in GROCERY_NAMES]
        self.goals = [Dummy(f"goal_{n.replace(' ', '_')}") for n in GROCERY_NAMES]

        # bottom shelf frame (everything goes to bottom shelf now)
        try:
            self.bottom_goals = Dummy("bottom_goals")
        except Exception:
            self.bottom_goals = None

        # (Optional) groceries spawn boundary
        self._has_grocery_boundary = False
        try:
            self._boundary_shape = Shape("groceries_boundary")
            self.groceries_boundary = SpawnBoundary([self._boundary_shape])
            self._has_grocery_boundary = True
        except Exception:
            self._boundary_shape = None
            self.groceries_boundary = None

        # Waypoints for cycle (14..20)
        self.wp_stage = Dummy(f"waypoint{WP_STAGE_FIXED}")  # 13
        self._stage_q0 = self.wp_stage.get_pose()[3:]

        self.wp_pick_app = Dummy(f"waypoint{WP_PICK_APPROACH}")  # 14
        self.wp_pick_grasp = Dummy(f"waypoint{WP_PICK_GRASP}")   # 15
        self.wp_pick_lift = Dummy(f"waypoint{WP_PICK_LIFT}")     # 16

        self.wp_transfer_up = Dummy(f"waypoint{WP_TRANSFER_UP}")      # 17
        self.wp_place_app = Dummy(f"waypoint{WP_PLACE_APPROACH}")     # 18
        self.wp_place_place = Dummy(f"waypoint{WP_PLACE_PLACE}")      # 19
        self.wp_place_retreat = Dummy(f"waypoint{WP_PLACE_RETREAT}")  # 20

        # --- Also grab 21..23 (spam pick) and 24..27 (mirrors of 17..20) ---
        try:
            self.wp2_pick_app = Dummy("waypoint21")
        except Exception:
            self.wp2_pick_app = None
        try:
            self.wp2_pick_grasp = Dummy("waypoint22")
        except Exception:
            self.wp2_pick_grasp = None
        try:
            self.wp2_pick_lift = Dummy("waypoint23")
        except Exception:
            self.wp2_pick_lift = None

        try:
            self.wp2_transfer_up = Dummy("waypoint24")
        except Exception:
            self.wp2_transfer_up = None
        try:
            self.wp2_place_app = Dummy("waypoint25")
        except Exception:
            self.wp2_place_app = None
        try:
            self.wp2_place_place = Dummy("waypoint26")
        except Exception:
            self.wp2_place_place = None
        try:
            self.wp2_place_retreat = Dummy("waypoint27")
        except Exception:
            self.wp2_place_retreat = None

        # Cache the original scene WP19 pose (used for both items; all bottom shelf)
        self._scene_wp19_pose = self.wp_place_place.get_pose()

    # -------------------- RLBench API --------------------
    def init_task(self) -> None:
        self._init_shoe_phase()
        self._init_grocery_phase()

        self.register_graspable_objects(
            cast(List[Object], [self.shoe1, self.shoe2] + self.groceries)
        )

        self.shoes_done = False
        self.register_waypoint_ability_start(SHOES_LAST_WP_IDX, self._start_groceries_phase)
        self.register_waypoint_ability_start(WP_STAGE_FIXED, self._on_stage_start)

        # PICK callbacks
        self.register_waypoint_ability_start(WP_PICK_APPROACH, self._on_pick_approach_start)
        self.register_waypoint_ability_start(WP_PICK_GRASP, self._on_pick_grasp_start)
        self.register_waypoint_ability_start(WP_PICK_LIFT, self._on_pick_lift_start)

        # TRANSFER + PLACE callbacks
        self.register_waypoint_ability_start(WP_TRANSFER_UP, self._on_transfer_up_start)
        self.register_waypoint_ability_start(WP_PLACE_APPROACH, self._on_place_approach_start)
        self.register_waypoint_ability_start(WP_PLACE_PLACE, self._on_place_place_start)
        self.register_waypoint_ability_start(WP_PLACE_RETREAT, self._on_place_retreat_start)

    def init_episode(self, index: int) -> List[str]:
        self.shoes_done = False
        self.groceries_placed = 0
        self._held_obj = None
        self._hard_locked = False

        if self._has_grocery_boundary and self.groceries_boundary is not None:
            self.groceries_boundary.clear()
            try:
                for g in self.groceries:
                    self.groceries_boundary.sample(g, min_distance=0.15)
            except BoundaryError:
                for g in self.groceries:
                    self.groceries_boundary.sample(g, ignore_collisions=True)

        self._prime_for_validation_first_item()

        # success only when BOTH shoes are in & ALL groceries detected.
        conds: List[Condition] = [
            DetectedCondition(self.shoe1, self.success_sensor_shoe),
            DetectedCondition(self.shoe2, self.success_sensor_shoe),
            DetectedSeveralCondition(
                cast(List[Object], self.groceries),
                self.success_sensor_grocery,
                self.groceries_to_place,
            ),
        ]
        self.register_success_conditions(conds)
        return [
            "put the shoes in the box, then put the groceries in the cupboard",
            "store the shoes, then store the groceries",
        ]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self):
        # keep everything stable; no random base yaw
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def is_static_workspace(self) -> bool:
        return True

    def boundary_root(self) -> Object:
        return Shape("boundary_root")

    # -------------------- Helpers --------------------
    def _current_index(self) -> int:
        return min(self.groceries_placed, self.groceries_to_place - 1)

    def _current_name(self) -> str:
        return GROCERY_NAMES[self._current_index()]

    # -------------------- Phase switch & staging --------------------
    def _start_groceries_phase(self, _) -> None:
        if self.shoes_done:
            return
        self.shoes_done = True
        # Repeat once per grocery (looping retained)
        self.register_waypoints_should_repeat(self._repeat_cycle)
        self._prime_runtime_for_current_item()

    def _on_stage_start(self, _) -> None:
        """Place WP13 above the current pick approach XY; tool-down orientation."""
        try:
            self._update_pick_approach_force()
            ax, ay, az, *_ = self.wp_pick_app.get_pose()
            _, _, tz, *_ = self.robot.arm.get_tip().get_pose()
            z = max(tz, az) + STAGE_CLEAR_DZ
            self.wp_stage.set_pose([ax, ay, z, *self._stage_q0])
        except Exception:
            tip = list(self.robot.arm.get_tip().get_pose())
            tip[2] += max(STAGE_CLEAR_DZ, 0.10)
            self.wp_stage.set_pose([tip[0], tip[1], tip[2], *self._stage_q0])

    def _repeat_cycle(self) -> bool:
        """Called by RLBench after WP20 to decide if we loop for another grocery."""
        if not self.shoes_done:
            return False

        # Finished placing the current one — unlock if needed.
        if self._held_obj is not None:
            self._unlock_from_tip(self._held_obj)

        # Advance to next grocery.
        self.groceries_placed += 1
        self._held_obj = None
        self._hard_locked = False

        # Done when we've placed them all.
        if self.groceries_placed >= self.groceries_to_place:
            return False

        # Keep the shoes waypoints inert so the planner doesn't wander back there.
        try:
            retreat_pose = self.wp_place_retreat.get_pose()
        except Exception:
            retreat_pose = self.robot.arm.get_tip().get_pose()
        try:
            for idx, d in self._pre_grocery_wp.items():  # 0..12
                d.set_pose(retreat_pose)
            self.wp_stage.set_pose(retreat_pose)  # 13
        except Exception:
            pass

        # Prime next item & ensure the gripper is open to approach.
        self._prime_runtime_for_current_item()
        try:
            self.robot.gripper.open()
        except Exception:
            pass

        # Tell RLBench to repeat the waypoint cycle.
        return True

    # -------------------- Priming helpers --------------------
    def _boundary_centre(self):
        if self._boundary_shape is not None:
            return self._boundary_shape.get_position()
        xs, ys, zs = zip(*(g.get_position() for g in self.groceries))
        return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))

    def _estimate_table_z(self) -> float:
        if self._boundary_shape is not None:
            return self._boundary_shape.get_position()[2]
        return min(g.get_position()[2] for g in self.groceries)

    def _prime_for_validation_first_item(self) -> None:
        """Set 13..20 before validation; set 21..23 from *spam*; mirror 17..20 -> 24..27."""
        # Soup (index 0) for 14..20
        i = 0
        gx, gy, gz, gqx, gqy, gqz, gqw = self.grasp_points[i].get_pose()
        self.wp_pick_app.set_pose([gx, gy, gz + PICK_APPROACH_DZ, gqx, gqy, gqz, gqw])
        self.wp_pick_grasp.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        self.wp_pick_lift.set_pose([gx, gy, gz + PICK_LIFT_DZ, gqx, gqy, gqz, gqw])

        # Spam (index 1) for 21..23 (make these *spam*-specific)
        self._prime_spam_pick_chain()

        self._update_transfer_up_force()  # sets 17
        # 18 = goal (dynamic per item)
        kx, ky, kz, kqx, kqy, kqz, kqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([kx, ky, kz, kqx, kqy, kqz, kqw])
        # 19 scene baseline (+ small X insert)
        self._update_place_place_force()
        # 20 = retreat back to 18
        self.wp_place_retreat.set_pose([kx, ky, kz, kqx, kqy, kqz, kqw])

        # Mirror 17..20 into 24..27 (exactly equal)
        self._mirror_17_to_24_chain()

        try:
            table_z = self._estimate_table_z()
            z = max(table_z + STAGE_Z_ABOVE_TABLE, gz + STAGE_Z_ABOVE_GRASP)
            self.wp_stage.set_pose([gx, gy, z, *self._stage_q0])
        except Exception:
            tip = list(self.robot.arm.get_tip().get_pose())
            tip[2] += max(STAGE_Z_ABOVE_TABLE, 0.30)
            self.wp_stage.set_pose([tip[0], tip[1], tip[2], *self._stage_q0])

    def _prime_spam_pick_chain(self) -> None:
        """Set 21,22,23 FROM spam_grasp_point (index 1), like 14,15,16 are from soup."""
        try:
            i = 1  # spam
            gx, gy, gz, gqx, gqy, gqz, gqw = self.grasp_points[i].get_pose()
            if self.wp2_pick_app is not None:
                self.wp2_pick_app.set_pose([gx, gy, gz + PICK_APPROACH_DZ, gqx, gqy, gqz, gqw])
            if self.wp2_pick_grasp is not None:
                self.wp2_pick_grasp.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
            if self.wp2_pick_lift is not None:
                self.wp2_pick_lift.set_pose([gx, gy, gz + PICK_LIFT_DZ, gqx, gqy, gqz, gqw])
        except Exception:
            pass

    def _prime_runtime_for_current_item(self) -> None:
        self._update_pick_approach_force()
        self._update_pick_grasp_force()
        self._update_pick_lift_force()
        self._update_place_chain_from_current_item()

    def _update_place_chain_from_current_item(self) -> None:
        """Recompute 17, 18, 19, and 20 (all bottom shelf now)."""
        self._update_transfer_up_force()  # 17

        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()

        # 18 = goal pose for this item
        self.wp_place_app.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])

        # 19 = scene XY/quat/Z (+ small X insert), identical for both items
        self._update_place_place_force()

        # 20 = retreat back to 18 (per item)
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])

        # Keep 24..27 mirrored to 17..20
        self._mirror_17_to_24_chain()

    # -------------------- HARD-LOCK HELPERS --------------------
    def _lock_to_tip(self, obj: Shape) -> None:
        """Parent the object to the tool tip so it can't fall if something opens."""
        try:
            self.robot.gripper.close()
            self.robot.gripper.grasp(obj)
            tip = self.robot.arm.get_tip()
            obj.set_parent(tip, keepInPlace=True)
            obj.set_dynamic(False)
            self._hard_locked = True
        except Exception:
            pass

    def _unlock_from_tip(self, obj: Shape) -> None:
        try:
            if self._hard_locked:
                obj.set_parent(None)
                self._hard_locked = False
                obj.set_dynamic(True)
            else:
                self._hard_locked = False
        except Exception:
            self._hard_locked = False

    # -------------------- Gripper keep-closed helper --------------------
    def _keep_closed_if_holding(self) -> None:
        """Re-close & re-grasp at each waypoint start to override any injected opens."""
        try:
            if self._held_obj is not None:
                self.robot.gripper.close()
                self.robot.gripper.grasp(self._held_obj)
        except Exception:
            pass

    # -------------------- Waypoint START handlers --------------------
    def _on_pick_approach_start(self, _) -> None:
        # Ensure it's OPEN while approaching WP14/15.
        self._update_pick_approach_force()
        try:
            self.robot.gripper.open()
        except Exception:
            pass

    def _on_pick_grasp_start(self, _) -> None:
        # Keep it open while moving to WP15.
        self._update_pick_grasp_force()
        try:
            self.robot.gripper.open()
        except Exception:
            pass

    def _on_pick_lift_start(self, _) -> None:
        # We are now AT WP15. Grasp + close + hard-lock, then lift.
        try:
            obj = self.groceries[self.groceries_placed]
            self._held_obj = obj
            self._lock_to_tip(obj)
        except Exception:
            pass
        self._update_pick_lift_force()
        self._update_place_chain_from_current_item()

    def _on_transfer_up_start(self, _) -> None:
        # Maintain closed during the vertical raise to WP17.
        self._update_transfer_up_force()
        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None:
                self._lock_to_tip(self._held_obj)  # idempotent
        except Exception:
            pass

    def _on_place_approach_start(self, _) -> None:
        # Cap twist about goal's Z on the 17->18 hop (to avoid whipping the can)
        i = self._current_index()

        # Goal pose (keep XYZ exactly)
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        q_goal = np.array([gqx, gqy, gqz, gqw], dtype=float)
        Rg = _quat_to_rot(gqx, gqy, gqz, gqw)

        # Current WP17 orientation
        _, _, _, qx17, qy17, qz17, qw17 = self.wp_transfer_up.get_pose()
        R17 = _quat_to_rot(qx17, qy17, qz17, qw17)

        # Express R17 in the goal frame
        R_rel = Rg.T @ R17

        # Extract twist around Z (in goal frame): yaw = atan2(r10, r00)
        twist = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
        twist_clamped = _clamp(twist, -TWIST_LIMIT_17_TO_18_RAD, TWIST_LIMIT_17_TO_18_RAD)

        # Local z-rotation quaternion (goal frame)
        q_local_z = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), twist_clamped)

        # New orientation = goal ⊗ local_z(twist_clamped)
        q_new = _quat_normalize(_quat_mul(q_goal, q_local_z))

        # Set WP18 to goal XYZ with limited-twist orientation
        self.wp_place_app.set_pose(
            [gx, gy, gz, float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3])]
        )

        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None:
                self._lock_to_tip(self._held_obj)
        except Exception:
            pass

        # Keep place chain mirrors updated as motion starts
        self._mirror_17_to_24_chain()

    def _on_place_place_start(self, _) -> None:
        # Prepare WP19 then release.
        self._update_place_place_force()
        try:
            if self._held_obj is not None:
                self._unlock_from_tip(self._held_obj)
            self.robot.gripper.release()
            self._held_obj = None
        except Exception:
            pass
        self._mirror_17_to_24_chain()

    def _on_place_retreat_start(self, _) -> None:
        self._update_place_retreat_force()
        self._mirror_17_to_24_chain()

    # -------------------- FORCE updaters --------------------
    def _update_pick_approach_force(self) -> None:
        i = self._current_index()
        x, y, z, qx, qy, qz, qw = self.grasp_points[i].get_pose()
        self.wp_pick_app.set_pose([x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])

    def _update_pick_grasp_force(self) -> None:
        i = self._current_index()
        self.wp_pick_grasp.set_pose(self.grasp_points[i].get_pose())

    def _update_pick_lift_force(self) -> None:
        i = self._current_index()
        x, y, z, qx, qy, qz, qw = self.grasp_points[i].get_pose()
        self.wp_pick_lift.set_pose([x, y, z + PICK_LIFT_DZ, qx, qy, qz, qw])

    def _update_transfer_up_force(self) -> None:
        """WP17 directly above WP16; same quat; clamp Z using the BOTTOM shelf (unified)."""
        x16, y16, z16, qx16, qy16, qz16, qw16 = self.wp_pick_lift.get_pose()

        # Always use bottom shelf now.
        if self.bottom_goals is not None:
            shelf_z = self.bottom_goals.get_position()[2]
        else:
            shelf_z = z16 + TRANSFER_MIN_RAISE

        dz = shelf_z - z16
        if dz < TRANSFER_MIN_RAISE:
            target_z = z16 + TRANSFER_MIN_RAISE
        elif dz > TRANSFER_MAX_RAISE:
            target_z = z16 + TRANSFER_MAX_RAISE
        else:
            target_z = shelf_z

        self.wp_transfer_up.set_pose([x16, y16, target_z, qx16, qy16, qz16, qw16])

        # Mirror place chain (24..27 := 17..20)
        self._mirror_17_to_24_chain()

    def _update_place_approach_force(self) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])  # 18 = goal
        self._mirror_17_to_24_chain()

    def _update_place_place_force(self) -> None:
        """WP19: always restore the scene pose (bottom shelf), but nudge +X slightly inside."""
        sx, sy, sz, sqx, sqy, sqz, sqw = self._scene_wp19_pose
        self.wp_place_place.set_pose([sx + WP19_INSERT_WORLD_DX, sy, sz, sqx, sqy, sqz, sqw])
        self._mirror_17_to_24_chain()

    def _update_place_retreat_force(self) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        self._mirror_17_to_24_chain()

    # -------------------- Mirroring helper (only 24..27) --------------------
    def _mirror_17_to_24_chain(self) -> None:
        """Keep waypoint24..27 EXACTLY equal to waypoint17..20 (poses)."""
        try:
            if self.wp2_transfer_up is not None:
                self.wp2_transfer_up.set_pose(self.wp_transfer_up.get_pose())
        except Exception:
            pass
        try:
            if self.wp2_place_app is not None:
                self.wp2_place_app.set_pose(self.wp_place_app.get_pose())
        except Exception:
            pass
        try:
            if self.wp2_place_place is not None:
                self.wp2_place_place.set_pose(self.wp_place_place.get_pose())
        except Exception:
            pass
        try:
            if self.wp2_place_retreat is not None:
                self.wp2_place_retreat.set_pose(self.wp_place_retreat.get_pose())
        except Exception:
            pass

    # -------------------- Shoes no-op --------------------
    def _noop_shoes_when_done(self, _) -> None:
        if not self.shoes_done:
            return
        # While repeating, keep shoes waypoints glued to current pose so they are no-ops
        try:
            pose = self.robot.arm.get_tip().get_pose()
            for idx, d in self._pre_grocery_wp.items():
                if idx < SHOES_LAST_WP_IDX:
                    d.set_pose(pose)
        except Exception:
            pass
