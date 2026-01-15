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


# -------------------- Waypoint layout (your spec) --------------------
# 0..4   : open box lid (scene-authored)
# 5..8   : pick shoe1 & hover over box (no release at 8)
# 9..11  : go back and place shoe1 where it came from (CLONES of 5,6,7)
# 12     : directly above 13 (same XY & orientation as 13, keep 12's Z) for a vertical 12→13 drop
# 13..15 : pick obstacle (trash) — 13 = grasp, 14/15 = vertical rises (straight line)
# 16     : hover over dustbin (scene handles release)
# 17..20 : pick shoe1 again; 20 hover over box (scene handles release)
# 21..24 : pick shoe2 again; 24 hover over box (scene handles release; 24 left AS-IS)
# 25     : PRE-APPROACH (new) — a safe high point above the current grocery grasp XY
# 26..28 : pick grocery #1 (soup): 26=approach (above), 27=grasp, 28=lift (above)
# 29     : transfer-up/stage (air stage above pick XY)
# 30..32 : place grocery #1 (approach, place, retreat)
# 33..35 : pick grocery #2 (spam): 33=approach (above), 34=grasp, 35=lift (above)
# 36     : transfer-up/stage for #2 (mirror of 29)
# 37..39 : place grocery #2 (approach, place, retreat)

# -------------------- indices for logic --------------------
SHOES_LAST_WP_IDX = 24  # end of shoes/obstacle phase

# Grocery cycle anchors (keep 26/27/28 semantics)
WP_PRE_APPROACH     = 25  # NEW: safe pre-approach/entry to pick area
WP_PICK_APPROACH    = 26
WP_PICK_GRASP       = 27
WP_PICK_LIFT        = 28
WP_STAGE_FIXED      = 29  # stage / transfer-up (air)
WP_PLACE_APPROACH   = 30
WP_PLACE_PLACE      = 31
WP_PLACE_RETREAT    = 32

# Second grocery
WP2_PICK_APPROACH   = 33
WP2_PICK_GRASP      = 34
WP2_PICK_LIFT       = 35
WP2_TRANSFER_UP     = 36
WP2_PLACE_APPROACH  = 37
WP2_PLACE_PLACE     = 38
WP2_PLACE_RETREAT   = 39

# ---- deltas (meters) ----
PICK_APPROACH_DZ = 0.20
PICK_LIFT_DZ = 0.22

# Transfer-up safety rails
TRANSFER_MIN_RAISE = 0.28
TRANSFER_MAX_RAISE = 0.40

# Stage safety
STAGE_Z_ABOVE_TABLE = 0.42
STAGE_Z_ABOVE_GRASP = 0.28
STAGE_CLEAR_DZ = 0.16

# Twist limit (goal-Z) for stage->place-approach hop
TWIST_LIMIT_17_TO_18_DEG = 70.0
TWIST_LIMIT_17_TO_18_RAD = np.deg2rad(TWIST_LIMIT_17_TO_18_DEG)

# Slight push inward at place
PLACE_INSERT_WORLD_DX = 0.03

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
    """Quaternion multiply (x,y,z,w) order: returns q1 ⊗ q2."""
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
    """Shoes + obstacle 0–24, then groceries 25–39.

    Changes here focus ONLY on 12/13/14/15 straight-line alignment and reliable grasp at 13.
    """

    # -------------------- SHOES / OBSTACLE PHASE --------------------
    def _init_shoe_phase(self) -> None:
        self.shoe1 = Shape("shoe1")
        self.shoe2 = Shape("shoe2")
        self.box_lid = Shape("box_lid")
        self.box_joint = Joint("box_joint")

        try:
            self.success_sensor_shoe = ProximitySensor("success_in_box")
        except Exception:
            self.success_sensor_shoe = None

        # cache waypoints 0..24 so we can keep them inert later (but we will SKIP 24)
        self._pre_grocery_wp: Dict[int, Dummy] = {}
        for i in range(SHOES_LAST_WP_IDX + 1):  # 0..24
            try:
                self._pre_grocery_wp[i] = Dummy(f"waypoint{i}")
            except Exception:
                pass
        for i in self._pre_grocery_wp.keys():
            self.register_waypoint_ability_start(i, self._noop_shoes_when_done)

        # Common dummies
        try:
            self.wp_hover_box = Dummy("waypoint8")  # your authored hover
        except Exception:
            self.wp_hover_box = None

        # Trash (try both common names)
        self.trash: Optional[Shape] = None
        for nm in ("trash", "rubbish"):
            try:
                self.trash = Shape(nm)
                break
            except Exception:
                continue

        # Hover above dustbin (release handled by scene)
        self.wp_hover_bin = None
        for name in ("dustbin_hover", "bin_hover", "waypoint16"):
            try:
                self.wp_hover_bin = Dummy(name)
                break
            except Exception:
                pass

        # Shoe grasp points
        try:
            self.shoe1_grasp = Dummy("shoe1_grasp_point")
        except Exception:
            self.shoe1_grasp = None
        try:
            self.shoe2_grasp = Dummy("shoe2_grasp_point")
        except Exception:
            self.shoe2_grasp = None

    # -------------------- GROCERIES PHASE --------------------
    def _init_grocery_phase(self) -> None:
        self.groceries: List[Shape] = [Shape(n.replace(" ", "_")) for n in GROCERY_NAMES]
        try:
            self.success_sensor_grocery = ProximitySensor("success")
        except Exception:
            self.success_sensor_grocery = None
        self.groceries_to_place = len(self.groceries)
        self.groceries_placed = 0

        # track held object + whether it's hard-locked (used only for groceries)
        self._held_obj: Optional[Shape] = None
        self._hard_locked: bool = False

        # grasp points and goals for soup/spam
        self.grasp_points = [Dummy(f"{n.replace(' ', '_')}_grasp_point") for n in GROCERY_NAMES]
        self.goals = [Dummy(f"goal_{n.replace(' ', '_')}") for n in GROCERY_NAMES]

        # Optional bottom shelf frame
        try:
            self.bottom_goals = Dummy("bottom_goals")
        except Exception:
            self.bottom_goals = None

        # Optional spawn boundary
        self._has_grocery_boundary = False
        try:
            self._boundary_shape = Shape("groceries_boundary")
            self.groceries_boundary = SpawnBoundary([self._boundary_shape])
            self._has_grocery_boundary = True
        except Exception:
            self._boundary_shape = None
            self.groceries_boundary = None

        # Waypoint handles (soup cycle)
        self.wp_pre_app = Dummy(f"waypoint{WP_PRE_APPROACH}")      # 25
        self.wp_pick_app = Dummy(f"waypoint{WP_PICK_APPROACH}")    # 26
        self.wp_pick_grasp = Dummy(f"waypoint{WP_PICK_GRASP}")     # 27
        self.wp_pick_lift = Dummy(f"waypoint{WP_PICK_LIFT}")       # 28
        self.wp_stage = Dummy(f"waypoint{WP_STAGE_FIXED}")         # 29
        self._stage_q0 = self.wp_stage.get_pose()[3:]
        self.wp_place_app = Dummy(f"waypoint{WP_PLACE_APPROACH}")  # 30
        self.wp_place_place = Dummy(f"waypoint{WP_PLACE_PLACE}")   # 31
        self.wp_place_retreat = Dummy(f"waypoint{WP_PLACE_RETREAT}")  # 32

        # Spam cycle handles
        try:
            self.wp2_pick_app = Dummy(f"waypoint{WP2_PICK_APPROACH}")   # 33
        except Exception:
            self.wp2_pick_app = None
        try:
            self.wp2_pick_grasp = Dummy(f"waypoint{WP2_PICK_GRASP}")    # 34
        except Exception:
            self.wp2_pick_grasp = None
        try:
            self.wp2_pick_lift = Dummy(f"waypoint{WP2_PICK_LIFT}")      # 35
        except Exception:
            self.wp2_pick_lift = None

        try:
            self.wp2_transfer_up = Dummy(f"waypoint{WP2_TRANSFER_UP}")  # 36
        except Exception:
            self.wp2_transfer_up = None
        try:
            self.wp2_place_app = Dummy(f"waypoint{WP2_PLACE_APPROACH}") # 37
        except Exception:
            self.wp2_place_app = None
        try:
            self.wp2_place_place = Dummy(f"waypoint{WP2_PLACE_PLACE}")  # 38
        except Exception:
            self.wp2_place_place = None
        try:
            self.wp2_place_retreat = Dummy(f"waypoint{WP2_PLACE_RETREAT}")  # 39
        except Exception:
            self.wp2_place_retreat = None

        # Cache scene place baselines (31 and 38) for insert
        self._scene_wp31_pose = self.wp_place_place.get_pose()
        if self.wp2_place_place is not None:
            self._scene_wp38_pose = self.wp2_place_place.get_pose()
        else:
            self._scene_wp38_pose = self._scene_wp31_pose

    # -------------------- RLBench API --------------------
    def init_task(self) -> None:
        self._init_shoe_phase()
        self._init_grocery_phase()

        # Register all graspables (include trash if present)
        objs: List[Object] = [self.shoe1, self.shoe2] + self.groceries
        if self.trash is not None:
            objs.append(self.trash)
        self.register_graspable_objects(cast(List[Object], objs))

        self.shoes_done = False

        # Start groceries after WP24 (end of shoes/obstacle phase)
        self.register_waypoint_ability_start(SHOES_LAST_WP_IDX, self._start_groceries_phase)
        self.register_waypoint_ability_start(WP_STAGE_FIXED, self._on_stage_start)

        # ---- TRASH pick callbacks (only around 12/13/14/15) ----
        self.register_waypoint_ability_start(13, self._on_trash_pick_start)   # close + grasp at 13
        self.register_waypoint_ability_start(14, self._on_trash_rise1_start)  # keep closed
        self.register_waypoint_ability_start(15, self._on_trash_rise2_start)  # keep closed

        # Grocery callbacks (guarded by shoes_done)
        self.register_waypoint_ability_start(WP_PICK_APPROACH, self._on_pick_approach_start)
        self.register_waypoint_ability_start(WP_PICK_GRASP, self._on_pick_grasp_start)
        self.register_waypoint_ability_start(WP_PICK_LIFT, self._on_pick_lift_start)
        self.register_waypoint_ability_start(WP_STAGE_FIXED, self._on_transfer_up_start)
        self.register_waypoint_ability_start(WP_PLACE_APPROACH, self._on_place_approach_start)
        self.register_waypoint_ability_start(WP_PLACE_PLACE, self._on_place_place_start)
        self.register_waypoint_ability_start(WP_PLACE_RETREAT, self._on_place_retreat_start)

    def init_episode(self, index: int) -> List[str]:
        self.shoes_done = False
        self.groceries_placed = 0
        self._held_obj = None
        self._hard_locked = False

        # Program shoes & obstacle (includes straight 12→13 and 13→14→15)
        self._program_shoes_and_obstacle_flow()

        # Boundary sample groceries (if available)
        if self._has_grocery_boundary and self.groceries_boundary is not None:
            self.groceries_boundary.clear()
            try:
                for g in self.groceries:
                    self.groceries_boundary.sample(g, min_distance=0.15)
            except BoundaryError:
                for g in self.groceries:
                    self.groceries_boundary.sample(g, ignore_collisions=True)

        # Pre-prime groceries (poses for 25..39), callbacks gated until shoes_done.
        self._prime_groceries_validation()

        # Success conditions (if sensors exist)
        conds: List[Condition] = []
        if self.success_sensor_shoe is not None:
            conds.extend([
                DetectedCondition(self.shoe1, self.success_sensor_shoe),
                DetectedCondition(self.shoe2, self.success_sensor_shoe),
            ])
        if self.success_sensor_grocery is not None:
            conds.append(
                DetectedSeveralCondition(
                    cast(List[Object], self.groceries),
                    self.success_sensor_grocery,
                    self.groceries_to_place,
                )
            )
        if conds:
            self.register_success_conditions(conds)

        return [
            "open the box, handle shoes and obstacle, then place groceries in the cupboard",
            "put both shoes in the box after removing obstacle, then store soup and spam",
        ]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def is_static_workspace(self) -> bool:
        return True

    def boundary_root(self) -> Object:
        return Shape("boundary_root")

    # -------------------- Shoes/Obstacle programming --------------------
    def _program_shoes_and_obstacle_flow(self) -> None:
        """5..12, 13..16, 17..24. 12 is aligned above 13; 13..15 are vertical & grasp at 13."""
        def set_wp(idx: int, pose):
            try:
                Dummy(f"waypoint{idx}").set_pose(pose)
            except Exception:
                pass

        def get_pose(idx: int):
            return Dummy(f"waypoint{idx}").get_pose()

        # ---- 5..8 : shoe1 pick & hover over box ----
        if self.shoe1_grasp is not None:
            x, y, z, qx, qy, qz, qw = self.shoe1_grasp.get_pose()
            set_wp(5, [x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])
            set_wp(6, [x, y, z,                 qx, qy, qz, qw])
            set_wp(7, [x, y, z + PICK_LIFT_DZ,  qx, qy, qz, qw])
        if self.wp_hover_box is not None:
            set_wp(8, self.wp_hover_box.get_pose())

        # ---- 9..11 : clones of 5..7 ----
        try:
            set_wp(9,  get_pose(5))
            set_wp(10, get_pose(6))
            set_wp(11, get_pose(7))
        except Exception:
            pass

        # ---- 12 : align directly above 13 — SAME XY & ORIENTATION as 13, keep 12's Z ----
        try:
            x13, y13, z13, qx13, qy13, qz13, qw13 = Dummy("waypoint13").get_pose()
            x12, y12, z12, _, _, _, _ = Dummy("waypoint12").get_pose()
            # snap 12 to (x13,y13) and orientation of 13, keep original z12 (above)
            Dummy("waypoint12").set_pose([x13, y13, z12, qx13, qy13, qz13, qw13])
        except Exception:
            pass

        # ---- 13..15 : straight line stack derived from 13 ----
        self._align_trash_chain_straight()  # sets 14 & 15 vertically above 13

        # ---- 16 : hover over dustbin (scene handles release) ----
        if self.wp_hover_bin is not None:
            set_wp(16, self.wp_hover_bin.get_pose())

        # ---- 17..20 : pick shoe1 again; 20 hover over box ----
        if self.shoe1_grasp is not None:
            x, y, z, qx, qy, qz, qw = self.shoe1_grasp.get_pose()
            set_wp(17, [x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])
            set_wp(18, [x, y, z,                 qx, qy, qz, qw])
            set_wp(19, [x, y, z + PICK_LIFT_DZ,  qx, qy, qz, qw])
        try:
            set_wp(20, get_pose(8))
        except Exception:
            pass

        # ---- 21..23 : pick shoe2 again; 24 hover (scene handles release) ----
        if self.shoe2_grasp is not None:
            x2, y2, z2, qx2, qy2, qz2, qw2 = self.shoe2_grasp.get_pose()
            set_wp(21, [x2, y2, z2 + PICK_APPROACH_DZ, qx2, qy2, qz2, qw2])
            set_wp(22, [x2, y2, z2,                 qx2, qy2, qz2, qw2])
            set_wp(23, [x2, y2, z2 + PICK_LIFT_DZ,  qx2, qy2, qz2, qw2])
        # do NOT touch waypoint24

    # -------------------- Start groceries after WP24 --------------------
    def _start_groceries_phase(self, _) -> None:
        if self.shoes_done:
            return
        self.shoes_done = True
        try:
            pose = self.robot.arm.get_tip().get_pose()
            for idx, d in self._pre_grocery_wp.items():
                if idx <= 23:  # do not touch 24
                    d.set_pose(pose)
        except Exception:
            pass
        self.register_waypoints_should_repeat(self._repeat_cycle)
        self._prime_runtime_for_current_item()

    # -------------------- Stage (runs ONLY after shoes_done) --------------------
    def _on_stage_start(self, _) -> None:
        if not self.shoes_done:
            return
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

    # -------------------- Repeat groceries for second item --------------------
    def _repeat_cycle(self) -> bool:
        if not self.shoes_done:
            return False
        if self._held_obj is not None:
            self._unlock_from_tip(self._held_obj)
        self.groceries_placed += 1
        self._held_obj = None
        self._hard_locked = False
        if self.groceries_placed >= self.groceries_to_place:
            return False
        self._mirror_29_to_36_chain()
        self._prime_runtime_for_current_item()
        try:
            self.robot.gripper.open()
        except Exception:
            pass
        return True

    # -------------------- TRASH: straight-line helper + callbacks --------------------
    def _align_trash_chain_straight(self) -> None:
        """13 is the grasp pose (scene-authored). Set 14/15 vertically above 13 with same XY & quat."""
        try:
            x13, y13, z13, qx13, qy13, qz13, qw13 = Dummy("waypoint13").get_pose()
            # 14: rise by PICK_LIFT_DZ
            Dummy("waypoint14").set_pose([x13, y13, z13 + PICK_LIFT_DZ, qx13, qy13, qz13, qw13])
            # 15: rise a bit more
            Dummy("waypoint15").set_pose([x13, y13, z13 + PICK_LIFT_DZ + STAGE_CLEAR_DZ,
                                          qx13, qy13, qz13, qw13])
        except Exception:
            pass

    def _on_trash_pick_start(self, _) -> None:
        """At 13: ensure 12 is straight-above, then close & grasp the trash (soft attach)."""
        self._align_trash_chain_straight()
        try:
            if self.trash is not None:
                # Make sure it's detachable & dynamic
                try:
                    self.trash.set_parent(None)
                except Exception:
                    pass
                try:
                    self.trash.set_dynamic(True)
                except Exception:
                    pass
                self.robot.gripper.close()
                self.robot.gripper.grasp(self.trash)
        except Exception:
            pass

    def _on_trash_rise1_start(self, _) -> None:
        try:
            if self.trash is not None:
                self.robot.gripper.close()
                self.robot.gripper.grasp(self.trash)
        except Exception:
            pass

    def _on_trash_rise2_start(self, _) -> None:
        try:
            if self.trash is not None:
                self.robot.gripper.close()
                self.robot.gripper.grasp(self.trash)
        except Exception:
            pass

    # -------------------- Groceries priming (poses only) --------------------
    def _boundary_centre(self):
        if self._boundary_shape is not None:
            return self._boundary_shape.get_position()
        xs, ys, zs = zip(*(g.get_position() for g in self.groceries))
        return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))

    def _estimate_table_z(self) -> float:
        if self._boundary_shape is not None:
            return self._boundary_shape.get_position()[2]
        return min(g.get_position()[2] for g in self.groceries)

    def _prime_groceries_validation(self) -> None:
        i = 0
        gx, gy, gz, gqx, gqy, gqz, gqw = self.grasp_points[i].get_pose()
        self._update_pre_approach_force_from_pose(gx, gy, gz, gqx, gqy, gqz, gqw)        # 25
        self.wp_pick_app.set_pose([gx, gy, gz + PICK_APPROACH_DZ, gqx, gqy, gqz, gqw])    # 26
        self.wp_pick_grasp.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])                      # 27
        self.wp_pick_lift.set_pose([gx, gy, gz + PICK_LIFT_DZ, gqx, gqy, gqz, gqw])        # 28

        self._update_transfer_up_force()  # 29
        kx, ky, kz, kqx, kqy, kqz, kqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([kx, ky, kz, kqx, kqy, kqz, kqw])                       # 30
        self._update_place_place_force()                                                   # 31
        self.wp_place_retreat.set_pose([kx, ky, kz, kqx, kqy, kqz, kqw])                   # 32

        i = 1
        gx, gy, gz, gqx, gqy, gqz, gqw = self.grasp_points[i].get_pose()
        if self.wp2_pick_app is not None:
            self.wp2_pick_app.set_pose([gx, gy, gz + PICK_APPROACH_DZ, gqx, gqy, gqz, gqw])  # 33
        if self.wp2_pick_grasp is not None:
            self.wp2_pick_grasp.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])                    # 34
        if self.wp2_pick_lift is not None:
            self.wp2_pick_lift.set_pose([gx, gy, gz + PICK_LIFT_DZ, gqx, gqy, gqz, gqw])      # 35

        self._mirror_29_to_36_chain()

        try:
            table_z = self._estimate_table_z()
            z = max(table_z + STAGE_Z_ABOVE_TABLE, gz + STAGE_Z_ABOVE_GRASP)
            self.wp_stage.set_pose([gx, gy, z, *self._stage_q0])
        except Exception:
            tip = list(self.robot.arm.get_tip().get_pose())
            tip[2] += max(STAGE_Z_ABOVE_TABLE, 0.30)
            self.wp_stage.set_pose([tip[0], tip[1], tip[2], *self._stage_q0])

    def _current_index(self) -> int:
        return min(self.groceries_placed, self.groceries_to_place - 1)

    def _current_name(self) -> str:
        return GROCERY_NAMES[self._current_index()]

    def _prime_runtime_for_current_item(self) -> None:
        self._update_pre_approach_force()
        self._update_pick_approach_force()
        self._update_pick_grasp_force()
        self._update_pick_lift_force()
        self._update_place_chain_from_current_item()

    # -------------------- Grocery callbacks (GUARDED) --------------------
    def _on_pick_approach_start(self, _) -> None:
        if not self.shoes_done:
            return
        self._update_pre_approach_force()
        self._update_pick_approach_force()
        try:
            self.robot.gripper.open()
        except Exception:
            pass

    def _on_pick_grasp_start(self, _) -> None:
        if not self.shoes_done:
            return
        self._update_pick_grasp_force()
        try:
            self.robot.gripper.open()
        except Exception:
            pass

    def _on_pick_lift_start(self, _) -> None:
        if not self.shoes_done:
            return
        try:
            obj = self.groceries[self.groceries_placed]
            self._held_obj = obj
            self._lock_to_tip(obj)
        except Exception:
            pass
        self._update_pick_lift_force()
        self._update_place_chain_from_current_item()

    def _on_transfer_up_start(self, _) -> None:
        if not self.shoes_done:
            return
        self._update_transfer_up_force()
        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None:
                self._lock_to_tip(self._held_obj)
        except Exception:
            pass

    def _on_place_approach_start(self, _) -> None:
        if not self.shoes_done:
            return
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        q_goal = np.array([gqx, gqy, gqz, gqw], dtype=float)
        Rg = _quat_to_rot(gqx, gqy, gqz, gqw)

        _, _, _, qx29, qy29, qz29, qw29 = self.wp_stage.get_pose()
        R29 = _quat_to_rot(qx29, qy29, qz29, qw29)
        R_rel = Rg.T @ R29
        twist = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
        twist_clamped = _clamp(twist, -TWIST_LIMIT_17_TO_18_RAD, TWIST_LIMIT_17_TO_18_RAD)
        q_local_z = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), twist_clamped)
        q_new = _quat_normalize(_quat_mul(q_goal, q_local_z))

        self.wp_place_app.set_pose([gx, gy, gz, float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3])])
        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None:
                self._lock_to_tip(self._held_obj)
        except Exception:
            pass
        self._mirror_29_to_36_chain()

    def _on_place_place_start(self, _) -> None:
        if not self.shoes_done:
            return
        self._update_place_place_force()
        try:
            if self._held_obj is not None:
                self._unlock_from_tip(self._held_obj)
            self.robot.gripper.release()
            self._held_obj = None
        except Exception:
            pass
        self._mirror_29_to_36_chain()

    def _on_place_retreat_start(self, _) -> None:
        if not self.shoes_done:
            return
        self._update_place_retreat_force()
        self._mirror_29_to_36_chain()

    # -------------------- FORCE updaters (groceries) --------------------
    def _update_pre_approach_force(self) -> None:
        i = self._current_index()
        x, y, z, qx, qy, qz, qw = self.grasp_points[i].get_pose()
        self._update_pre_approach_force_from_pose(x, y, z, qx, qy, qz, qw)

    def _update_pre_approach_force_from_pose(self, x, y, z, qx, qy, qz, qw) -> None:
        z_pre = z + max(STAGE_Z_ABOVE_GRASP, PICK_APPROACH_DZ + 0.05)
        try:
            self.wp_pre_app.set_pose([x, y, z_pre, qx, qy, qz, qw])  # 25
        except Exception:
            try:
                Dummy("waypoint25").set_pose([x, y, z_pre, qx, qy, qz, qw])
            except Exception:
                pass

    def _update_pick_approach_force(self) -> None:
        i = self._current_index()
        x, y, z, qx, qy, qz, qw = self.grasp_points[i].get_pose()
        self.wp_pick_app.set_pose([x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])  # 26

    def _update_pick_grasp_force(self) -> None:
        i = self._current_index()
        self.wp_pick_grasp.set_pose(self.grasp_points[i].get_pose())             # 27

    def _update_pick_lift_force(self) -> None:
        i = self._current_index()
        x, y, z, qx, qy, qz, qw = self.grasp_points[i].get_pose()
        self.wp_pick_lift.set_pose([x, y, z + PICK_LIFT_DZ, qx, qy, qz, qw])     # 28

    def _update_transfer_up_force(self) -> None:
        xL, yL, zL, qxL, qyL, qzL, qwL = self.wp_pick_lift.get_pose()  # from 28

        if self.bottom_goals is not None:
            shelf_z = self.bottom_goals.get_position()[2]
        else:
            shelf_z = zL + TRANSFER_MIN_RAISE

        dz = shelf_z - zL
        if dz < TRANSFER_MIN_RAISE:
            target_z = zL + TRANSFER_MIN_RAISE
        elif dz > TRANSFER_MAX_RAISE:
            target_z = zL + TRANSFER_MAX_RAISE
        else:
            target_z = shelf_z

        self.wp_stage.set_pose([xL, yL, target_z, qxL, qyL, qzL, qwL])  # 29
        self._mirror_29_to_36_chain()

    def _update_place_chain_from_current_item(self) -> None:
        self._update_transfer_up_force()  # 29
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])  # 30
        self._update_place_place_force()                               # 31
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])  # 32
        self._mirror_29_to_36_chain()

    def _update_place_place_force(self) -> None:
        sx, sy, sz, sqx, sqy, sqz, sqw = self._scene_wp31_pose
        self.wp_place_place.set_pose([sx + PLACE_INSERT_WORLD_DX, sy, sz, sqx, sqy, sqz, sqw])
        if self.wp2_place_place is not None and self._scene_wp38_pose is not None:
            sx2, sy2, sz2, sqx2, sqy2, sqz2, sqw2 = self._scene_wp38_pose
            self.wp2_place_place.set_pose([sx2 + PLACE_INSERT_WORLD_DX, sy2, sz2, sqx2, sqy2, sqz2, sqw2])

    def _update_place_retreat_force(self) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        self._mirror_29_to_36_chain()

    # -------------------- Mirror helper (36..39 := 29..32) --------------------
    def _mirror_29_to_36_chain(self) -> None:
        try:
            if self.wp2_transfer_up is not None:
                self.wp2_transfer_up.set_pose(self.wp_stage.get_pose())
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

    # -------------------- Hard-lock helpers (for groceries only) --------------------
    def _lock_to_tip(self, obj: Shape) -> None:
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

    def _keep_closed_if_holding(self) -> None:
        try:
            if self._held_obj is not None:
                self.robot.gripper.close()
                self.robot.gripper.grasp(self._held_obj)
        except Exception:
            pass

    # -------------------- Keep shoes WPs inert after shoes_done --------------------
    def _noop_shoes_when_done(self, _) -> None:
        if not self.shoes_done:
            return
        try:
            pose = self.robot.arm.get_tip().get_pose()
            for idx, d in self._pre_grocery_wp.items():
                if idx <= 23:  # do not touch 24
                    d.set_pose(pose)
        except Exception:
            pass
