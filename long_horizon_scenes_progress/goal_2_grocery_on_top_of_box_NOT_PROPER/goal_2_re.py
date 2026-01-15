from typing import List, Dict, cast, Optional, Iterable, Any
import numpy as np

from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.cartesian_path import CartesianPath  # scene path for lid/open

from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedCondition,
    DetectedSeveralCondition,
    Condition,
)
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.exceptions import BoundaryError


# -------------------- indexing --------------------
# Scene has: 0..7  = NEW soup-from-box → cupboard
#            8..35 = original flow (shoes + groceries) [shifted by +4, logic unchanged]
SHIFT = 8  # everything that was 4..31 becomes 8..35

# ---- original logical waypoint indices (unshifted, for readability) ----
SHOES_LAST_WP_IDX = 12

WP_STAGE_FIXED     = 13
WP_PICK_APPROACH   = 14
WP_PICK_GRASP      = 15
WP_PICK_LIFT       = 16
WP_TRANSFER_UP     = 17
WP_PLACE_APPROACH  = 18
WP_PLACE_PLACE     = 19
WP_PLACE_RETREAT   = 20

# Spam pick
WP2_PICK_APPROACH  = 21
WP2_PICK_GRASP     = 22
WP2_PICK_LIFT      = 23

# Mirrors of 17..20
WP2_TRANSFER_UP    = 24
WP2_PLACE_APPROACH = 25
WP2_PLACE_PLACE    = 26
WP2_PLACE_RETREAT  = 27

# ---- runtime (actual scene indices 8..35) ----
R_SHOES_LAST_WP_IDX = SHOES_LAST_WP_IDX + SHIFT    # 20
R_WP_STAGE_FIXED     = WP_STAGE_FIXED + SHIFT       # 21

R_WP_PICK_APPROACH   = WP_PICK_APPROACH + SHIFT     # 22
R_WP_PICK_GRASP      = WP_PICK_GRASP + SHIFT        # 23
R_WP_PICK_LIFT       = WP_PICK_LIFT + SHIFT         # 24
R_WP_TRANSFER_UP     = WP_TRANSFER_UP + SHIFT       # 25
R_WP_PLACE_APPROACH  = WP_PLACE_APPROACH + SHIFT    # 26
R_WP_PLACE_PLACE     = WP_PLACE_PLACE + SHIFT       # 27
R_WP_PLACE_RETREAT   = WP_PLACE_RETREAT + SHIFT     # 28

R_WP2_PICK_APPROACH  = WP2_PICK_APPROACH + SHIFT    # 29
R_WP2_PICK_GRASP     = WP2_PICK_GRASP + SHIFT       # 30
R_WP2_PICK_LIFT      = WP2_PICK_LIFT + SHIFT        # 31

R_WP2_TRANSFER_UP    = WP2_TRANSFER_UP + SHIFT      # 32
R_WP2_PLACE_APPROACH = WP2_PLACE_APPROACH + SHIFT   # 33
R_WP2_PLACE_PLACE    = WP2_PLACE_PLACE + SHIFT      # 34
R_WP2_PLACE_RETREAT  = WP2_PLACE_RETREAT + SHIFT    # 35

# ---- NEW 0..7: soup on top of box → cupboard ----
OBS0_PICK_APP      = 0
OBS1_PICK_GRASP    = 1
OBS2_PICK_LIFT     = 2
OBS3_TRANSFER_UP   = 3
OBS4_PLACE_APP     = 4
OBS5_PLACE_PLACE   = 5
OBS6_PLACE_RETREAT = 6
OBS7_ROBOT_START   = 7  # dynamically set to the robot start pose

# ---- deltas (meters) ----
PICK_APPROACH_DZ = 0.20
PICK_LIFT_DZ     = 0.22

TRANSFER_MIN_RAISE = 0.28
TRANSFER_MAX_RAISE = 0.40

STAGE_Z_ABOVE_TABLE = 0.42
STAGE_Z_ABOVE_GRASP = 0.28
STAGE_CLEAR_DZ      = 0.16

TWIST_LIMIT_17_TO_18_DEG = 70.0
TWIST_LIMIT_17_TO_18_RAD = np.deg2rad(TWIST_LIMIT_17_TO_18_DEG)

WP19_INSERT_WORLD_DX = 0.03

# We will place soup_0 first (phase 0..7) and then in groceries phase place soup (non _0) and spam
GROCERY_NAMES = ["soup", "spam"]


# -------------------- math helpers --------------------
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


def _rot_apply(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    return R @ v


# -------------------- small finders (make names robust) --------------------
def _try_get_dummy(names: List[str]) -> Optional[Dummy]:
    for nm in names:
        try:
            return Dummy(nm)
        except Exception:
            pass
    return None


def _try_get_shape(names: List[str]) -> Optional[Shape]:
    for nm in names:
        try:
            return Shape(nm)
        except Exception:
            pass
    return None


# -------------------- task --------------------
class Goal2Re(Task):
    # -------------------- Gripper helpers (actuate wrapper) --------------------
    def _gripper_open(self, velocity: float = 0.2) -> None:
        try:
            self.robot.gripper.actuate(1.0, velocity)  # 1.0 == open (Robotiq85)
        except Exception:
            pass

    def _gripper_close(self, velocity: float = 0.2) -> None:
        try:
            self.robot.gripper.actuate(0.0, velocity)  # 0.0 == close
        except Exception:
            pass

    # ---- small utility for robust move_to_pose signatures ----
    def _move_to_pose(self, pose: Iterable[float], v: float = 0.35, a: float = 0.30) -> None:
        """Call move_to_pose with either positional or keyword args depending on build."""
        try:
            self.robot.arm.move_to_pose(list(pose), v, a)
        except TypeError:
            self.robot.arm.move_to_pose(list(pose), max_velocity=v, max_acceleration=a)

    # -------------------- 0..7: Soup-on-box phase --------------------
    def _init_obstacle_phase(self) -> None:
        def _safe_dummy(name: str) -> Optional[Dummy]:
            try:
                return Dummy(name)
            except Exception:
                return None

        self.wp0_app   = _safe_dummy(f"waypoint{OBS0_PICK_APP}")
        self.wp1_grasp = _safe_dummy(f"waypoint{OBS1_PICK_GRASP}")
        self.wp2_lift  = _safe_dummy(f"waypoint{OBS2_PICK_LIFT}")
        self.wp3_up    = _safe_dummy(f"waypoint{OBS3_TRANSFER_UP}")
        self.wp4_app   = _safe_dummy(f"waypoint{OBS4_PLACE_APP}")
        self.wp5_place = _safe_dummy(f"waypoint{OBS5_PLACE_PLACE}")
        self.wp6_ret   = _safe_dummy(f"waypoint{OBS6_PLACE_RETREAT}")
        self.wp7_start = _safe_dummy(f"waypoint{OBS7_ROBOT_START}")

        # cache robot start pose -> waypoint7
        try:
            self._robot_start_pose = self.robot.arm.get_tip().get_pose()
            if self.wp7_start is not None:
                self.wp7_start.set_pose(self._robot_start_pose)
        except Exception:
            self._robot_start_pose = None

        # ---- FIX: prefer the soup on the lid (soup_0), try several name variants.
        # Grasp dummy (you have 'soup_grasp_point0' in your scene)
        self._soup_grasp_dummy = _try_get_dummy([
            "soup_grasp_point_0",  # underscore variant
            "soup_grasp_point0",   # your scene
            "soup_0_grasp_point",
            "soup0_grasp_point",
        ])

        # Physical soup object to grasp first
        self._soup_obj_top = _try_get_shape([
            "soup_0",   # preferred
            "soup0",
            "soup (0)",
        ])

        self._held_obstacle: Optional[Shape] = None

        # Register ability starts for 0..6
        self.register_waypoint_ability_start(OBS0_PICK_APP,      self._on_obs_pick_approach_start)
        self.register_waypoint_ability_start(OBS1_PICK_GRASP,    self._on_obs_pick_grasp_start)
        self.register_waypoint_ability_start(OBS2_PICK_LIFT,     self._on_obs_pick_lift_start)
        self.register_waypoint_ability_start(OBS3_TRANSFER_UP,   self._on_obs_transfer_up_start)
        self.register_waypoint_ability_start(OBS4_PLACE_APP,     self._on_obs_place_approach_start)
        self.register_waypoint_ability_start(OBS5_PLACE_PLACE,   self._on_obs_place_place_start)
        self.register_waypoint_ability_start(OBS6_PLACE_RETREAT, self._on_obs_place_retreat_start)

    # ---- 0..7 pose updaters ----
    def _obs_update_pick_chain(self) -> None:
        """Set 0,1,2 from the soup_0 grasp dummy (or fallback) every time."""
        if self._soup_grasp_dummy is None:
            return
        x, y, z, qx, qy, qz, qw = self._soup_grasp_dummy.get_pose()
        if self.wp0_app is not None:
            self.wp0_app.set_pose([x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])
        if self.wp1_grasp is not None:
            self.wp1_grasp.set_pose([x, y, z, qx, qy, qz, qw])
        if self.wp2_lift is not None:
            self.wp2_lift.set_pose([x, y, z + PICK_LIFT_DZ, qx, qy, qz, qw])

    def _obs_update_transfer_up(self) -> None:
        """Waypoint3 trivially feasible: equal to waypoint2 (with safe tiny bump)."""
        if self.wp2_lift is None or self.wp3_up is None:
            return
        x20, y20, z20, qx20, qy20, qz20, qw20 = self.wp2_lift.get_pose()
        bump = 0.08
        z_target = z20 if (bump < TRANSFER_MIN_RAISE or bump > TRANSFER_MAX_RAISE) else z20 + bump
        self.wp3_up.set_pose([x20, y20, z_target, qx20, qy20, qz20, qw20])

    def _obs_update_place_chain(self) -> None:
        """Waypoints 4..6 from soup_0's cupboard goal like 26,27,28."""
        try:
            gx, gy, gz, gqx, gqy, gqz, gqw = Dummy("goal_soup_0").get_pose()
        except Exception:
            try:
                gx, gy, gz, gqx, gqy, gqz, gqw = Dummy("goal_soup").get_pose()
            except Exception:
                bg = Dummy("bottom_goals")
                gx, gy, gz = bg.get_pose()[:3]
                gqx, gqy, gqz, gqw = bg.get_pose()[3:]
        if self.wp4_app is not None:
            self.wp4_app.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        if self.wp5_place is not None:
            try:
                scene = Dummy(f"waypoint{R_WP_PLACE_PLACE}")  # 27 in runtime
                sx, sy, sz, sqx, sqy, sqz, sqw = scene.get_pose()
                self.wp5_place.set_pose([sx + WP19_INSERT_WORLD_DX, sy, sz, sqx, sqy, sqz, sqw])
            except Exception:
                self.wp5_place.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        if self.wp6_ret is not None:
            self.wp6_ret.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])

    # ---- 0..7 handlers ----
    def _on_obs_pick_approach_start(self, _) -> None:
        self._obs_update_pick_chain()
        self._gripper_open()

    def _on_obs_pick_grasp_start(self, _) -> None:
        self._obs_update_pick_chain()
        self._gripper_open()

    def _on_obs_pick_lift_start(self, _) -> None:
        try:
            self._gripper_close()
            target = self._soup_obj_top
            if target is not None:
                self.robot.gripper.grasp(target)
                self._held_obstacle = target
        except Exception:
            pass
        self._obs_update_pick_chain()
        self._obs_update_transfer_up()
        self._obs_update_place_chain()

    def _on_obs_transfer_up_start(self, _) -> None:
        self._obs_update_transfer_up()
        try:
            if self._held_obstacle is not None:
                self._gripper_close()
                self.robot.gripper.grasp(self._held_obstacle)
        except Exception:
            pass

    def _on_obs_place_approach_start(self, _) -> None:
        self._obs_update_place_chain()
        try:
            if self._held_obstacle is not None:
                self._gripper_close()
                self.robot.gripper.grasp(self._held_obstacle)
        except Exception:
            pass

    def _on_obs_place_place_start(self, _) -> None:
        self._obs_update_place_chain()
        try:
            self.robot.gripper.release()
            self._held_obstacle = None
        except Exception:
            pass

    def _on_obs_place_retreat_start(self, _) -> None:
        self._obs_update_place_chain()

    # -------------------- 8..20: Shoes + Lid open --------------------
    def _init_shoe_phase(self) -> None:
        self.shoe1 = Shape("shoe1")
        self.shoe2 = Shape("shoe2")
        self.box_lid = Shape("box_lid")
        self.box_joint = Joint("box_joint")
        self.success_sensor_shoe = ProximitySensor("success_in_box")

        self._pre_grocery_wp: Dict[int, Optional[Dummy]] = {}
        for i in range(SHIFT, R_WP_STAGE_FIXED):  # 8..20
            try:
                self._pre_grocery_wp[i] = Dummy(f"waypoint{i}")
            except Exception:
                self._pre_grocery_wp[i] = None

        # Load lid path (kept under box_lid). We try common names.
        self._lid_path: Optional[CartesianPath] = None
        for nm in ("lid_open_path", "waypoint10", "waypoint10_path", "wp10_path", "box_open_path"):
            try:
                self._lid_path = CartesianPath(nm)
                break
            except Exception:
                self._lid_path = None

        # Pin waypoint10 to waypoint9 initially so 9→10 is trivial
        try:
            wp9 = self._pre_grocery_wp.get(9)
            wp10 = self._pre_grocery_wp.get(10)
            if wp9 is not None and wp10 is not None:
                wp10.set_pose(wp9.get_pose())
        except Exception:
            pass

        # Register ability starts:
        for idx in range(SHIFT, R_WP_STAGE_FIXED):
            if idx == 9:
                self.register_waypoint_ability_start(idx, self._on_lid_contact_pose_start)
            elif idx == 10:
                self.register_waypoint_ability_start(idx, self._on_lid_grasp_and_open_start)
            else:
                self.register_waypoint_ability_start(idx, self._noop_shoes_when_done)

    def _on_lid_contact_pose_start(self, _) -> None:
        self._gripper_open()

    def _nudge_from_dummy_along_local_x(self, d: Dummy, distance: float = 0.018) -> List[float]:
        x, y, z, qx, qy, qz, qw = d.get_pose()
        R = _quat_to_rot(qx, qy, qz, qw)
        offset = _rot_apply(R, np.array([distance, 0.0, 0.0], dtype=float))
        nx, ny, nz = x + float(offset[0]), y + float(offset[1]), z + float(offset[2])
        return [nx, ny, nz, qx, qy, qz, qw]

    def _follow_lid_path_or_hinge(self) -> None:
        try:
            if self._lid_path is not None:
                if hasattr(self.robot.arm, "move_along_path"):
                    self.robot.arm.move_along_path(self._lid_path, max_vel=0.35, accel=0.30)
                    return
                if hasattr(self.robot.arm, "follow_path"):
                    try:
                        self.robot.arm.follow_path(self._lid_path, max_vel=0.35, accel=0.30)
                    except TypeError:
                        self.robot.arm.follow_path(self._lid_path, max_velocity=0.35, max_acceleration=0.30)
                    return
        except Exception:
            pass

        # Fallback: hinge sweep
        try:
            if self.box_joint is not None:
                try:
                    cyclic, interval = self.box_joint.get_joint_interval()
                    lower = float(interval[0])
                    rng = float(interval[1])
                    upper = lower + rng
                except Exception:
                    lower, upper = -np.pi, np.pi
                target_open_rad = np.deg2rad(95.0)
                target = float(_clamp(target_open_rad, lower, upper))
                current = float(self.box_joint.get_joint_position())
                steps = 80
                for k in range(steps):
                    a = current + (target - current) * (k + 1) / steps
                    try:
                        self.box_joint.set_joint_position(a)
                    except Exception:
                        break
        except Exception:
            pass

    def _on_lid_grasp_and_open_start(self, _) -> None:
        try:
            wp9 = self._pre_grocery_wp.get(9)
            if wp9 is not None:
                nudge_pose = self._nudge_from_dummy_along_local_x(wp9, distance=0.018)
                self._move_to_pose(nudge_pose, v=0.35, a=0.30)
        except Exception:
            pass

        try:
            self._gripper_close()
            if self.box_lid is not None:
                self.robot.gripper.grasp(self.box_lid)
        except Exception:
            pass

        self._follow_lid_path_or_hinge()

        try:
            self.robot.gripper.release()
        except Exception:
            pass

        try:
            wp10 = self._pre_grocery_wp.get(10)
            if wp10 is not None:
                tip_pose = self.robot.arm.get_tip().get_pose()
                wp10.set_pose(tip_pose)
        except Exception:
            pass

        self._noop_shoes_when_done(_)

    # -------------------- 21..35: Groceries --------------------
    def _init_grocery_phase(self) -> None:
        self.groceries: List[Shape] = []
        for n in GROCERY_NAMES:
            try:
                self.groceries.append(Shape(n.replace(" ", "_")))
            except Exception:
                pass
        self.success_sensor_grocery = ProximitySensor("success")
        self.groceries_to_place = len(self.groceries)
        self.groceries_placed = 0

        self._held_obj: Optional[Shape] = None
        self._hard_locked: bool = False

        self.grasp_points: List[Dummy] = []
        for name in ("soup_grasp_point", "spam_grasp_point"):
            d = _try_get_dummy([name, name.replace("_point", "")])
            if d is not None:
                self.grasp_points.append(d)

        self.goals: List[Dummy] = []
        for name in ("goal_soup", "goal_spam"):
            d = _try_get_dummy([name, "bottom_goals"])
            if d is not None:
                self.goals.append(d)

        try:
            self.bottom_goals = Dummy("bottom_goals")
        except Exception:
            self.bottom_goals = None

        self._has_grocery_boundary = False
        try:
            self._boundary_shape = Shape("groceries_boundary")
            self.groceries_boundary = SpawnBoundary([self._boundary_shape])
            self._has_grocery_boundary = True
        except Exception:
            self._boundary_shape = None
            self.groceries_boundary = None

        self.wp_stage = Dummy(f"waypoint{R_WP_STAGE_FIXED}")          # 21
        self._stage_q0 = self.wp_stage.get_pose()[3:]

        self.wp_pick_app = Dummy(f"waypoint{R_WP_PICK_APPROACH}")     # 22
        self.wp_pick_grasp = Dummy(f"waypoint{R_WP_PICK_GRASP}")      # 23
        self.wp_pick_lift = Dummy(f"waypoint{R_WP_PICK_LIFT}")        # 24

        self.wp_transfer_up = Dummy(f"waypoint{R_WP_TRANSFER_UP}")    # 25
        self.wp_place_app = Dummy(f"waypoint{R_WP_PLACE_APPROACH}")   # 26
        self.wp_place_place = Dummy(f"waypoint{R_WP_PLACE_PLACE}")    # 27
        self.wp_place_retreat = Dummy(f"waypoint{R_WP_PLACE_RETREAT}")# 28

        # Spam pick 29..31
        try:
            self.wp2_pick_app = Dummy(f"waypoint{R_WP2_PICK_APPROACH}")  # 29
        except Exception:
            self.wp2_pick_app = None
        try:
            self.wp2_pick_grasp = Dummy(f"waypoint{R_WP2_PICK_GRASP}")   # 30
        except Exception:
            self.wp2_pick_grasp = None
        try:
            self.wp2_pick_lift = Dummy(f"waypoint{R_WP2_PICK_LIFT}")     # 31
        except Exception:
            self.wp2_pick_lift = None

        # Mirrors 32..35
        try:
            self.wp2_transfer_up = Dummy(f"waypoint{R_WP2_TRANSFER_UP}")      # 32
        except Exception:
            self.wp2_transfer_up = None
        try:
            self.wp2_place_app = Dummy(f"waypoint{R_WP2_PLACE_APPROACH}")     # 33
        except Exception:
            self.wp2_place_app = None
        try:
            self.wp2_place_place = Dummy(f"waypoint{R_WP2_PLACE_PLACE}")      # 34
        except Exception:
            self.wp2_place_place = None
        try:
            self.wp2_place_retreat = Dummy(f"waypoint{R_WP2_PLACE_RETREAT}")  # 35
        except Exception:
            self.wp2_place_retreat = None

        self._scene_wp19_pose = self.wp_place_place.get_pose()

    # -------------------- RLBench API --------------------
    def init_task(self) -> None:
        self._init_obstacle_phase()   # 0..7
        self._init_shoe_phase()       # 8..20
        self._init_grocery_phase()    # 21..35

        graspables: List[Object] = cast(List[Object], [self.shoe1, self.shoe2] + self.groceries)
        self.register_graspable_objects(graspables)

        self.shoes_done = False

        self.register_waypoint_ability_start(R_SHOES_LAST_WP_IDX, self._start_groceries_phase)
        self.register_waypoint_ability_start(R_WP_STAGE_FIXED, self._on_stage_start)

        self.register_waypoint_ability_start(R_WP_PICK_APPROACH, self._on_pick_approach_start)
        self.register_waypoint_ability_start(R_WP_PICK_GRASP, self._on_pick_grasp_start)
        self.register_waypoint_ability_start(R_WP_PICK_LIFT, self._on_pick_lift_start)

        self.register_waypoint_ability_start(R_WP_TRANSFER_UP, self._on_transfer_up_start)
        self.register_waypoint_ability_start(R_WP_PLACE_APPROACH, self._on_place_approach_start)
        self.register_waypoint_ability_start(R_WP_PLACE_PLACE, self._on_place_place_start)
        self.register_waypoint_ability_start(R_WP_PLACE_RETREAT, self._on_place_retreat_start)

        self.register_waypoints_should_repeat(self._repeat_cycle)

        # Ensure 0..2 track soup_0 from the start
        self._obs_update_pick_chain()
        self._obs_update_transfer_up()
        self._obs_update_place_chain()

    def init_episode(self, index: int) -> List[str]:
        self.shoes_done = False
        self.groceries_placed = 0
        self._held_obj = None
        self._hard_locked = False
        self._held_obstacle = None

        if getattr(self, "_has_grocery_boundary", False) and self.groceries_boundary is not None:
            self.groceries_boundary.clear()
            try:
                for g in self.groceries:
                    self.groceries_boundary.sample(g, min_distance=0.15)
            except BoundaryError:
                for g in self.groceries:
                    self.groceries_boundary.sample(g, ignore_collisions=True)

        self._prime_for_validation_first_item()

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
            "do soup-on-box first (0–7), open lid & place shoes (8–20), then place remaining groceries (21–35)",
        ]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self):
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

        # Restage WP21 safely from current tip pose/orientation
        try:
            tip = list(self.robot.arm.get_tip().get_pose())
            tip[2] += max(STAGE_CLEAR_DZ, 0.12)
            self.wp_stage.set_pose(tip)
        except Exception:
            pass

        self.register_waypoints_should_repeat(self._repeat_cycle)
        self._prime_runtime_for_current_item()

    def _on_stage_start(self, _) -> None:
        try:
            self._update_pick_approach_force()
            ax, ay, az, *_ = self.wp_pick_app.get_pose()
            sx, sy, sz, qx, qy, qz, qw = self.wp_stage.get_pose()
            z = max(sz, az)
            self.wp_stage.set_pose([ax, ay, z, qx, qy, qz, qw])
        except Exception:
            tip = list(self.robot.arm.get_tip().get_pose())
            tip[2] += max(STAGE_CLEAR_DZ, 0.10)
            self.wp_stage.set_pose(tip)

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

        try:
            retreat_pose = self.wp_place_retreat.get_pose()
        except Exception:
            retreat_pose = self.robot.arm.get_tip().get_pose()
        try:
            for idx, d in self._pre_grocery_wp.items():  # 8..20
                if d is not None:
                    d.set_pose(retreat_pose)
            self.wp_stage.set_pose(retreat_pose)  # 21
        except Exception:
            pass

        self._prime_runtime_for_current_item()
        self._gripper_open()
        return True

    # -------------------- Priming helpers --------------------
    def _boundary_centre(self):
        if getattr(self, "_boundary_shape", None) is not None:
            return self._boundary_shape.get_position()
        xs, ys, zs = zip(*(g.get_position() for g in self.groceries))
        return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))

    def _estimate_table_z(self) -> float:
        if getattr(self, "_boundary_shape", None) is not None:
            return self._boundary_shape.get_position()[2]
        return min(g.get_position()[2] for g in self.groceries)

    def _prime_for_validation_first_item(self) -> None:
        i = 0  # first grocery = soup (non _0)
        gx, gy, gz, gqx, gqy, gqz, gqw = self.grasp_points[i].get_pose()
        self.wp_pick_app.set_pose([gx, gy, gz + PICK_APPROACH_DZ, gqx, gqy, gqz, gqw])  # 22
        self.wp_pick_grasp.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])                   # 23
        self.wp_pick_lift.set_pose([gx, gy, gz + PICK_LIFT_DZ, gqx, gqy, gqz, gqw])     # 24

        self._prime_spam_pick_chain()

        self._update_transfer_up_force()  # 25
        kx, ky, kz, kqx, kqy, kqz, kqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([kx, ky, kz, kqx, kqy, kqz, kqw])                    # 26
        self._update_place_place_force()                                                 # 27
        self.wp_place_retreat.set_pose([kx, ky, kz, kqx, kqy, kqz, kqw])                 # 28

        self._mirror_17_to_24_chain()

        try:
            table_z = self._estimate_table_z()
            z = max(table_z + STAGE_Z_ABOVE_TABLE, gz + STAGE_Z_ABOVE_GRASP)
            sx, sy, _, qx, qy, qz, qw = self.wp_stage.get_pose()
            self.wp_stage.set_pose([gx, gy, z, qx, qy, qz, qw])
        except Exception:
            tip = list(self.robot.arm.get_tip().get_pose())
            tip[2] += max(STAGE_Z_ABOVE_TABLE, 0.30)
            self.wp_stage.set_pose(tip)

    def _prime_spam_pick_chain(self) -> None:
        try:
            i = 1
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
        self._update_transfer_up_force()  # 25
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])  # 26
        self._update_place_place_force()                               # 27
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])  # 28
        self._mirror_17_to_24_chain()

    # -------------------- HARD-LOCK HELPERS (groceries) --------------------
    def _lock_to_tip(self, obj: Shape) -> None:
        try:
            self._gripper_close()
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
        try:
            if self._held_obj is not None:
                self._gripper_close()
                self.robot.gripper.grasp(self._held_obj)
        except Exception:
            pass

    # -------------------- Waypoint START handlers (groceries) --------------------
    def _on_pick_approach_start(self, _) -> None:
        self._update_pick_approach_force()
        self._gripper_open()

    def _on_pick_grasp_start(self, _) -> None:
        self._update_pick_grasp_force()
        self._gripper_open()

    def _on_pick_lift_start(self, _) -> None:
        try:
            obj = self.groceries[self.groceries_placed]
            self._held_obj = obj
            self._lock_to_tip(obj)
        except Exception:
            pass
        self._update_pick_lift_force()
        self._update_place_chain_from_current_item()

    def _on_transfer_up_start(self, _) -> None:
        self._update_transfer_up_force()
        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None:
                self._lock_to_tip(self._held_obj)
        except Exception:
            pass

    def _on_place_approach_start(self, _) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        q_goal = np.array([gqx, gqy, gqz, gqw], dtype=float)
        Rg = _quat_to_rot(gqx, gqy, gqz, gqw)

        _, _, _, qx21, qy21, qz21, qw21 = self.wp_transfer_up.get_pose()
        R21 = _quat_to_rot(qx21, qy21, qz21, qw21)

        R_rel = Rg.T @ R21
        twist = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
        twist_clamped = _clamp(twist, -TWIST_LIMIT_17_TO_18_RAD, TWIST_LIMIT_17_TO_18_RAD)
        def _axis_angle_quat_local_z(angle: float) -> np.ndarray:
            s = np.sin(angle * 0.5)
            return np.array([0.0 * s, 0.0 * s, 1.0 * s, np.cos(angle * 0.5)], dtype=float)
        q_local_z = _axis_angle_quat_local_z(twist_clamped)
        q_new = _quat_normalize(_quat_mul(q_goal, q_local_z))

        self.wp_place_app.set_pose(
            [gx, gy, gz, float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3])]
        )

        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None:
                self._lock_to_tip(self._held_obj)
        except Exception:
            pass

        self._mirror_17_to_24_chain()

    def _on_place_place_start(self, _) -> None:
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

    # -------------------- FORCE updaters (groceries) --------------------
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
        x20, y20, z20, qx20, qy20, qz20, qw20 = self.wp_pick_lift.get_pose()

        if getattr(self, "bottom_goals", None) is not None:
            shelf_z = self.bottom_goals.get_position()[2]
        else:
            shelf_z = z20 + TRANSFER_MIN_RAISE

        dz = shelf_z - z20
        if dz < TRANSFER_MIN_RAISE:
            target_z = z20 + TRANSFER_MIN_RAISE
        elif dz > TRANSFER_MAX_RAISE:
            target_z = z20 + TRANSFER_MAX_RAISE
        else:
            target_z = shelf_z

        self.wp_transfer_up.set_pose([x20, y20, target_z, qx20, qy20, qz20, qw20])
        self._mirror_17_to_24_chain()

    def _update_place_approach_force(self) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_app.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        self._mirror_17_to_24_chain()

    def _update_place_place_force(self) -> None:
        sx, sy, sz, sqx, sqy, sqz, sqw = self._scene_wp19_pose
        self.wp_place_place.set_pose([sx + WP19_INSERT_WORLD_DX, sy, sz, sqx, sqy, sqz, sqw])
        self._mirror_17_to_24_chain()

    def _update_place_retreat_force(self) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        self._mirror_17_to_24_chain()

    # -------------------- Mirroring helper (32..35) --------------------
    def _mirror_17_to_24_chain(self) -> None:
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
        try:
            if self._held_obstacle is not None:
                self.robot.gripper.release()
                self._held_obstacle = None
        except Exception:
            pass

        if not self.shoes_done:
            return
        try:
            pose = self.robot.arm.get_tip().get_pose()
            for idx, d in self._pre_grocery_wp.items():
                if d is not None:
                    d.set_pose(pose)
        except Exception:
            pass
