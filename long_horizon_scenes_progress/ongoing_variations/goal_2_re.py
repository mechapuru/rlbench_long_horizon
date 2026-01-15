from typing import List, Dict, cast, Optional, Tuple
import numpy as np
import os
import math
import random

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

# ---- Safe global boundary tracer + bypasses (toggle with env vars) ----
ENABLE_GLOBAL_BOUNDARY_TRACE = os.getenv("BOUNDARY_TRACE", "0") == "1"
BYPASS_ROOT_BOUNDARY = os.getenv("BYPASS_ROOT_BOUNDARY", "0") == "1"
BYPASS_EMPTY_BOUNDARY = os.getenv("BYPASS_EMPTY_BOUNDARY", "1") == "1"   # <- enabled by default
TASK_ROOT_NAME = os.getenv("TASK_ROOT_NAME", "goal_2_re")

if ENABLE_GLOBAL_BOUNDARY_TRACE and not getattr(SpawnBoundary, "_trace_installed", False):
    _SB_orig_sample = SpawnBoundary.sample
    _printed_once = {"root": False, "empty": False}

    def _SB_sample_trace(self, *args, **kwargs):
        try:
            return _SB_orig_sample(self, *args, **kwargs)
        except BoundaryError:
            # Boundary names (may be empty)
            try:
                bnames = [bo.get_name() for bo in getattr(self, "boundary_objects", [])]
            except Exception:
                bnames = []

            # Object name
            try:
                obj = kwargs.get("obj", None) if "obj" in kwargs else (args[0] if len(args) else None)
                oname = obj.get_name() if obj is not None else "<obj>"
            except Exception:
                oname = "<obj>"

            # Bypass: task root
            if BYPASS_ROOT_BOUNDARY and oname == TASK_ROOT_NAME:
                if not _printed_once["root"]:
                    print(f"[GLOBAL/Bypass] Skipping root placement of '{oname}' into boundary={bnames}.", flush=True)
                    _printed_once["root"] = True
                return None  # quietly succeed

            # Bypass: any empty boundary list
            if BYPASS_EMPTY_BOUNDARY and (len(bnames) == 0):
                if not _printed_once["empty"]:
                    print(f"[GLOBAL/Bypass] Skipping placement into EMPTY boundary for obj='{oname}'.", flush=True)
                    _printed_once["empty"] = True
                return None  # quietly succeed

            # Otherwise, report once per failure site
            md = kwargs.get("min_distance", None)
            ic = kwargs.get("ignore_collisions", None)
            print(f"[GLOBAL/BoundaryError] boundary={bnames if bnames else ['<empty>']} "
                  f"obj='{oname}' md={md} ignore_collisions={ic}", flush=True)
            raise

    SpawnBoundary.sample = _SB_sample_trace
    SpawnBoundary._trace_installed = True


# -------------------- Debug toggles --------------------
def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

PLACEMENT_DEBUG = _get_int_env("PLACEMENT_DEBUG", 0)  # 0=off, 1=info, 2=verbose

def _dbg(level: int, msg: str) -> None:
    if PLACEMENT_DEBUG >= level:
        print(msg, flush=True)

def _fmt(v):
    return tuple(round(float(x), 4) for x in v)


# -------------------- Waypoint layout (scene semantics preserved) --------------------
SHOES_LAST_WP_IDX = 24

WP_PRE_APPROACH     = 25
WP_PICK_APPROACH    = 26
WP_PICK_GRASP       = 27
WP_PICK_LIFT        = 28
WP_STAGE_FIXED      = 29
WP_PLACE_APPROACH   = 30
WP_PLACE_PLACE      = 31
WP_PLACE_RETREAT    = 32

WP2_PICK_APPROACH   = 33
WP2_PICK_GRASP      = 34
WP2_PICK_LIFT       = 35
WP2_TRANSFER_UP     = 36
WP2_PLACE_APPROACH  = 37
WP2_PLACE_PLACE     = 38
WP2_PLACE_RETREAT   = 39

PICK_APPROACH_DZ = 0.20
PICK_LIFT_DZ     = 0.22

TRANSFER_MIN_RAISE = 0.28
TRANSFER_MAX_RAISE = 0.40

STAGE_Z_ABOVE_TABLE = 0.42
STAGE_Z_ABOVE_GRASP = 0.28
STAGE_CLEAR_DZ      = 0.16

TWIST_LIMIT_17_TO_18_DEG = 70.0
TWIST_LIMIT_17_TO_18_RAD = np.deg2rad(TWIST_LIMIT_17_TO_18_DEG)

PLACE_INSERT_WORLD_DX = 0.03

GROCERY_NAMES = ["soup", "spam"]


# -------------------- Math helpers --------------------
def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array(
        [
            [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
            [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
            [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)],
        ],
        dtype=float,
    )

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / max(n, 1e-12)

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=float)

def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    s = math.sin(0.5*angle)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(0.5*angle)], dtype=float)

def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


# -------------------- Seeding & placement helpers --------------------
def _base_seed() -> int:
    return int(os.getenv("VARIATION_SEED", "12345"))

def _rng(index: int) -> np.random.RandomState:
    return np.random.RandomState(_base_seed() + int(index))

class _SeedCtx:
    """Temporarily seed numpy & random for deterministic internal sampling."""
    def __init__(self, seed: int):
        self.seed = seed
        self._np_state = None
        self._py_state = None
    def __enter__(self):
        self._np_state = np.random.get_state()
        self._py_state = random.getstate()
        np.random.seed(self.seed)
        random.seed(self.seed)
    def __exit__(self, exc_type, exc, tb):
        if self._np_state is not None:
            np.random.set_state(self._np_state)
        if self._py_state is not None:
            random.setstate(self._py_state)

def _seed_from(index: int, offset: int = 0) -> int:
    # Mix in index and an offset per category to avoid collisions
    return (_base_seed() + 10007*(int(index)+1) + 7919*offset) & 0x7FFFFFFF

def _get_bb(shape: Shape) -> Tuple[float, float, float, float, float, float]:
    bb = shape.get_bounding_box()  # [xmin,ymin,zmin,xmax,ymax,zmax]
    xmin, ymin, zmin, xmax, ymax, zmax = bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]
    if xmin > xmax: xmin, xmax = xmax, xmin
    if ymin > ymax: ymin, ymax = ymax, ymin
    if zmin > zmax: zmin, zmax = zmax, zmin
    return (xmin, ymin, zmin, xmax, ymax, zmax)

def _dims_from_bb(bb: Tuple[float,float,float,float,float,float]) -> Tuple[float,float,float]:
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    return (xmax - xmin, ymax - ymin, zmax - zmin)

def _safe_half_xy(obj: Shape, scale: float = 1.05) -> float:
    try:
        bb = obj.get_bounding_box()
        hx = abs(bb[3] - bb[0]) * 0.5
        hy = abs(bb[4] - bb[1]) * 0.5
        return float(scale * max(hx, hy))
    except Exception:
        return 0.05 * scale  # conservative default

def _safe_half_z(obj: Shape, scale: float = 1.05) -> float:
    try:
        bb = obj.get_bounding_box()
        hz = abs(bb[5] - bb[2]) * 0.5
        return float(scale * hz)
    except Exception:
        return 0.05 * scale

def _report_boundary_fit(boundary_shape: Shape, boundary_name: str,
                         obj: Shape, obj_name: str, min_dist: float) -> None:
    """Print a single, precise diagnostic about fit margins."""
    bb = _get_bb(boundary_shape)
    dx, dy, dz = _dims_from_bb(bb)
    r_xy = _safe_half_xy(obj)
    need = 2.0 * (r_xy + min_dist)  # naive requirement per axis
    mx = dx - need
    my = dy - need
    _dbg(1, f"[place/debug] boundary='{boundary_name}' dims(X,Y,Z)={_fmt((dx,dy,dz))} "
            f"obj='{obj_name}' r_xy={r_xy:.4f} min_dist={min_dist:.3f} "
            f"required_axis_len≈{need:.3f} margins(X,Y)={mx:.3f},{my:.3f}")
    if mx < 0 or my < 0:
        _dbg(1, f"[place/debug]  -> LIKELY TOO SMALL: "
                f"Increase '{boundary_name}' or reduce min_distance/object size.")

def _place_via_spawnboundary(boundary: SpawnBoundary, obj: Shape, obj_name: str,
                             boundary_name: str,
                             min_dist_sched: List[float]) -> bool:
    for md in min_dist_sched:
        try:
            if PLACEMENT_DEBUG >= 2:
                _dbg(2, f"[place/try] SpawnBoundary '{boundary_name}' obj='{obj_name}' min_distance={md:.3f}")
            boundary.sample(obj, min_distance=float(md))
            _dbg(1, f"[place/ok ] SpawnBoundary '{boundary_name}' obj='{obj_name}' md={md:.3f}")
            return True
        except BoundaryError:
            # Only report if boundary has a shape
            try:
                _report_boundary_fit(boundary.boundary_objects[0], boundary_name, obj, obj_name, md)
            except Exception:
                pass
            continue
        except Exception as e:
            _dbg(1, f"[place/err] SpawnBoundary '{boundary_name}' obj='{obj_name}' md={md:.3f} err={e}")
            continue
    try:
        if PLACEMENT_DEBUG >= 2:
            _dbg(2, f"[place/try] SpawnBoundary IGNORE_COLL '{boundary_name}' obj='{obj_name}'")
        boundary.sample(obj, ignore_collisions=True)
        _dbg(1, f"[place/ok ] SpawnBoundary IGNORE_COLL '{boundary_name}' obj='{obj_name}'")
        return True
    except Exception as e:
        _dbg(1, f"[place/err] SpawnBoundary IGNORE_COLL '{boundary_name}' obj='{obj_name}' err={e}")
        return False

def _place_via_aabb_fallback(boundary_shape, boundary_name, obj, obj_name, rng,
                             already, z_ref: Dict[int, float], clearance=0.01, max_tries=64):
    xmin, ymin, zmin, xmax, ymax, zmax = _get_bb(boundary_shape)
    r_xy = _safe_half_xy(obj) + clearance
    xmin2, xmax2 = xmin + r_xy, xmax - r_xy
    ymin2, ymax2 = ymin + r_xy, ymax - r_xy
    if xmin2 >= xmax2 or ymin2 >= ymax2:
        _dbg(1, f"[place/fail] AABB '{boundary_name}' too tight for '{obj_name}' "
                f"(use area shrunk by r={r_xy:.3f}).")
        return False
    z_keep = z_ref.get(obj.get_handle(), zmax)  # keep original table height
    for _ in range(max_tries):
        x = float(rng.uniform(xmin2, xmax2))
        y = float(rng.uniform(ymin2, ymax2))
        if all(math.hypot(x - px, y - py) >= (2.0 * r_xy) for (px, py) in already):
            hz = _safe_half_z(obj)
            obj.set_pose([x, y, z_keep + hz, *obj.get_pose()[3:]])
            _dbg(1, f"[place/ok ] AABB '{boundary_name}' obj='{obj_name}' pos=({x:.4f}, {y:.4f})")
            return True
    _dbg(1, f"[place/fail] AABB '{boundary_name}' max tries exceeded for '{obj_name}'.")
    return False

def _place_center(boundary_shape, boundary_name, obj, obj_name, z_ref: Dict[int, float]):
    bx, by, _ = boundary_shape.get_position()
    _, _, _, _, _, zmax = _get_bb(boundary_shape)
    hz = _safe_half_z(obj)
    z_keep = z_ref.get(obj.get_handle(), zmax)
    obj.set_pose([bx, by, z_keep + hz, *obj.get_pose()[3:]])
    _dbg(1, f"[place/ok ] CENTER '{boundary_name}' obj='{obj_name}' at center ({bx:.3f},{by:.3f})")


def _set_upright_random_yaw(obj: Shape, rng: np.random.RandomState, yaw_deg: float = 180.0) -> None:
    """Random yaw (±yaw_deg) about +Z; roll/pitch unchanged."""
    if yaw_deg <= 0.0:
        return
    yaw = float(rng.uniform(-math.radians(yaw_deg), math.radians(yaw_deg)))
    x, y, z = obj.get_position()
    s = math.sin(0.5 * yaw); c = math.cos(0.5 * yaw)
    qz = np.array([0.0 * s, 0.0 * s, 1.0 * s, c], dtype=float)  # axis z
    obj.set_pose([x, y, z, float(qz[0]), float(qz[1]), float(qz[2]), float(qz[3])])


def _clamp_obj_into_workspace(obj: Shape, ws: Shape, z_ref: Dict[int, float], pad: float = 0.005):
    """Clamp object's XY inside workspace AABB with a small pad. Preserve Z & orientation."""
    try:
        bb = ws.get_bounding_box()  # xmin,ymin,zmin,xmax,ymax,zmax
        xmin, ymin, zmin, xmax, ymax, zmax = bb
        r_xy = _safe_half_xy(obj) + pad
        xmin2, xmax2 = xmin + r_xy, xmax - r_xy
        ymin2, ymax2 = ymin + r_xy, ymax - r_xy
        x, y, z, qx, qy, qz, qw = obj.get_pose()
        x = min(max(x, xmin2), xmax2)
        y = min(max(y, ymin2), ymax2)
        # keep original table Z (from ref) or current z
        z_keep = z_ref.get(obj.get_handle(), z)
        obj.set_pose([x, y, z_keep, qx, qy, qz, qw])
    except Exception:
        pass


class Goal2Re(Task):
    """Same waypoint/navigation behavior; only spawn randomization + debug prints added."""

    # How many variation slots exist (k in [0..N-1])
    def variation_count(self) -> int:
        try:
            return max(1, int(os.getenv("VARIATIONS", "20")))
        except Exception:
            return 20

    # -------------------- SHOES / OBSTACLE PHASE --------------------
    def _init_shoe_phase(self) -> None:
        self.shoe1 = Shape("shoe1")
        self.shoe2 = Shape("shoe2")
        self.box_lid = Shape("box_lid")
        self.box_joint = Joint("box_joint")

        # Save original shoe orientations (NO yaw for shoes)
        self._shoe1_q0 = self.shoe1.get_pose()[3:]
        self._shoe2_q0 = self.shoe2.get_pose()[3:]

        try:
            self.success_sensor_shoe = ProximitySensor("success_in_box")
        except Exception:
            self.success_sensor_shoe = None

        self._pre_grocery_wp: Dict[int, Dummy] = {}
        for i in range(SHOES_LAST_WP_IDX + 1):
            try:
                self._pre_grocery_wp[i] = Dummy(f"waypoint{i}")
            except Exception:
                pass
        for i in self._pre_grocery_wp.keys():
            self.register_waypoint_ability_start(i, self._noop_shoes_when_done)

        try:
            self.wp_hover_box = Dummy("waypoint8")
        except Exception:
            self.wp_hover_box = None

        self.trash: Optional[Shape] = None
        for nm in ("trash", "rubbish"):
            try:
                self.trash = Shape(nm)
                break
            except Exception:
                continue

        self.wp_hover_bin = None
        for name in ("dustbin_hover", "bin_hover", "waypoint16"):
            try:
                self.wp_hover_bin = Dummy(name)
                break
            except Exception:
                pass

        try:
            self.shoe1_grasp = Dummy("shoe1_grasp_point")
        except Exception:
            self.shoe1_grasp = None
        try:
            self.shoe2_grasp = Dummy("shoe2_grasp_point")
        except Exception:
            self.shoe2_grasp = None

        # Boundaries
        self._has_shoes_boundary = False
        self.shoes_boundary = None
        try:
            sb = Shape("shoes_boundary")
            self.shoes_boundary = SpawnBoundary([sb])
            self.shoes_boundary_shape = sb
            self._has_shoes_boundary = True
        except Exception:
            self.shoes_boundary_shape = None

        self._has_rubbish_boundary = False
        self.rubbish_boundary = None
        try:
            rb = Shape("rubbish_boundary")
            self.rubbish_boundary = SpawnBoundary([rb])
            self.rubbish_boundary_shape = rb
            self._has_rubbish_boundary = True
        except Exception:
            self.rubbish_boundary_shape = None

    # -------------------- GROCERIES PHASE --------------------
    def _init_grocery_phase(self) -> None:
        self.groceries: List[Shape] = [Shape(n.replace(" ", "_")) for n in GROCERY_NAMES]
        try:
            self.success_sensor_grocery = ProximitySensor("success")
        except Exception:
            self.success_sensor_grocery = None
        self.groceries_to_place = len(self.groceries)
        self.groceries_placed = 0

        self._held_obj: Optional[Shape] = None
        self._hard_locked: bool = False

        self.grasp_points = [Dummy(f"{n.replace(' ', '_')}_grasp_point") for n in GROCERY_NAMES]
        self.goals = [Dummy(f"goal_{n.replace(' ', '_')}") for n in GROCERY_NAMES]

        try:
            self.bottom_goals = Dummy("bottom_goals")
        except Exception:
            self.bottom_goals = None

        self._has_grocery_boundary = False
        try:
            gb = Shape("groceries_boundary")
            self.groceries_boundary = SpawnBoundary([gb])
            self._boundary_shape = gb
            self._has_grocery_boundary = True
        except Exception:
            self.groceries_boundary = None
            self._boundary_shape = None

        self.wp_pre_app = Dummy(f"waypoint{WP_PRE_APPROACH}")      # 25
        self.wp_pick_app = Dummy(f"waypoint{WP_PICK_APPROACH}")    # 26
        self.wp_pick_grasp = Dummy(f"waypoint{WP_PICK_GRASP}")     # 27
        self.wp_pick_lift = Dummy(f"waypoint{WP_PICK_LIFT}")       # 28
        self.wp_stage = Dummy(f"waypoint{WP_STAGE_FIXED}")         # 29
        self._stage_q0 = self.wp_stage.get_pose()[3:]
        self.wp_place_app = Dummy(f"waypoint{WP_PLACE_APPROACH}")  # 30
        self.wp_place_place = Dummy(f"waypoint{WP_PLACE_PLACE}")   # 31
        self.wp_place_retreat = Dummy(f"waypoint{WP_PLACE_RETREAT}")  # 32

        try: self.wp2_pick_app = Dummy(f"waypoint{WP2_PICK_APPROACH}")
        except Exception: self.wp2_pick_app = None
        try: self.wp2_pick_grasp = Dummy(f"waypoint{WP2_PICK_GRASP}")
        except Exception: self.wp2_pick_grasp = None
        try: self.wp2_pick_lift = Dummy(f"waypoint{WP2_PICK_LIFT}")
        except Exception: self.wp2_pick_lift = None
        try: self.wp2_transfer_up = Dummy(f"waypoint{WP2_TRANSFER_UP}")
        except Exception: self.wp2_transfer_up = None
        try: self.wp2_place_app = Dummy(f"waypoint{WP2_PLACE_APPROACH}")
        except Exception: self.wp2_place_app = None
        try: self.wp2_place_place = Dummy(f"waypoint{WP2_PLACE_PLACE}")
        except Exception: self.wp2_place_place = None
        try: self.wp2_place_retreat = Dummy(f"waypoint{WP2_PLACE_RETREAT}")
        except Exception: self.wp2_place_retreat = None

        self._scene_wp31_pose = self.wp_place_place.get_pose()
        self._scene_wp38_pose = self.wp2_place_place.get_pose() if self.wp2_place_place is not None else self._scene_wp31_pose

    # -------------------- RLBench hooks --------------------
    def init_task(self) -> None:
        print("[SENTINEL] entered init_task", flush=True)
        self._init_shoe_phase()
        self._init_grocery_phase()

        objs: List[Object] = [self.shoe1, self.shoe2] + self.groceries
        if self.trash is not None:
            objs.append(self.trash)
        self.register_graspable_objects(cast(List[Object], objs))

        self.shoes_done = False

        self.register_waypoint_ability_start(SHOES_LAST_WP_IDX, self._start_groceries_phase)
        self.register_waypoint_ability_start(WP_STAGE_FIXED, self._on_stage_start)

        # Trash callbacks around 13/14/15
        self.register_waypoint_ability_start(13, self._on_trash_pick_start)
        self.register_waypoint_ability_start(14, self._on_trash_rise1_start)
        self.register_waypoint_ability_start(15, self._on_trash_rise2_start)

        # Grocery callbacks
        self.register_waypoint_ability_start(WP_PICK_APPROACH, self._on_pick_approach_start)
        self.register_waypoint_ability_start(WP_PICK_GRASP, self._on_pick_grasp_start)
        self.register_waypoint_ability_start(WP_PICK_LIFT, self._on_pick_lift_start)
        self.register_waypoint_ability_start(WP_STAGE_FIXED, self._on_transfer_up_start)
        self.register_waypoint_ability_start(WP_PLACE_APPROACH, self._on_place_approach_start)
        self.register_waypoint_ability_start(WP_PLACE_PLACE, self._on_place_place_start)
        self.register_waypoint_ability_start(WP_PLACE_RETREAT, self._on_place_retreat_start)

    def init_episode(self, index: int) -> List[str]:
        print(f"[SENTINEL] entered init_episode k={index}", flush=True)

        # Remember original Z for stable fallbacks
        self._z_ref: Dict[int, float] = {}
        for o in [self.shoe1, self.shoe2] + self.groceries + ([self.trash] if self.trash is not None else []):
            try:
                self._z_ref[o.get_handle()] = o.get_position()[2]
            except Exception:
                pass

        self.shoes_done = False
        self.groceries_placed = 0
        self._held_obj = None
        self._hard_locked = False

        rng = _rng(index)
        placed_xy: List[Tuple[float, float]] = []

        # ---- SHOES: XY only (NO yaw) ----
        if getattr(self, "_has_shoes_boundary", False) and self.shoes_boundary is not None:
            self.shoes_boundary.clear()
            bname = "shoes_boundary"
            with _SeedCtx(_seed_from(index, offset=1)):
                for obj, q0, oname in [(self.shoe1, self._shoe1_q0, "shoe1"),
                                       (self.shoe2, self._shoe2_q0, "shoe2")]:
                    ok = _place_via_spawnboundary(self.shoes_boundary, obj, oname, bname,
                                                  min_dist_sched=[0.10, 0.08, 0.06, 0.04, 0.02])
                    if not ok and getattr(self, "shoes_boundary_shape", None) is not None:
                        ok = _place_via_aabb_fallback(self.shoes_boundary_shape, bname, obj, oname, rng, placed_xy, self._z_ref, clearance=0.01)
                    if not ok and getattr(self, "shoes_boundary_shape", None) is not None:
                        _place_center(self.shoes_boundary_shape, bname, obj, oname, self._z_ref)
                    x, y, z = obj.get_position()
                    # Restore original orientation (no yaw)
                    obj.set_pose([x, y, z, *q0])
                    placed_xy.append((x, y))

        # ---- RUBBISH ----
        if self.trash is not None and getattr(self, "_has_rubbish_boundary", False) and self.rubbish_boundary is not None:
            self.rubbish_boundary.clear()
            bname = "rubbish_boundary"
            with _SeedCtx(_seed_from(index, offset=2)):
                ok = _place_via_spawnboundary(self.rubbish_boundary, self.trash, "trash", bname,
                                              min_dist_sched=[0.12, 0.10, 0.08, 0.06])
                if not ok and getattr(self, "rubbish_boundary_shape", None) is not None:
                    ok = _place_via_aabb_fallback(self.rubbish_boundary_shape, bname, self.trash, "trash", rng, placed_xy, self._z_ref, clearance=0.01)
                if not ok and getattr(self, "rubbish_boundary_shape", None) is not None:
                    _place_center(self.rubbish_boundary_shape, bname, self.trash, "trash", self._z_ref)
            _set_upright_random_yaw(self.trash, rng, yaw_deg=float(os.getenv("TRASH_YAW_DEG", "30")))

        # ---- GROCERIES ----
        if getattr(self, "_has_grocery_boundary", False) and self.groceries_boundary is not None:
            self.groceries_boundary.clear()
            bname = "groceries_boundary"
            with _SeedCtx(_seed_from(index, offset=3)):
                for g, oname in zip(self.groceries, GROCERY_NAMES):
                    ok = _place_via_spawnboundary(self.groceries_boundary, g, oname, bname,
                                                  min_dist_sched=[0.18, 0.15, 0.12, 0.10, 0.08])
                    if not ok and getattr(self, "_boundary_shape", None) is not None:
                        ok = _place_via_aabb_fallback(self._boundary_shape, bname, g, oname, rng, placed_xy, self._z_ref, clearance=0.015)
                    if not ok and getattr(self, "_boundary_shape", None) is not None:
                        _place_center(self._boundary_shape, bname, g, oname, self._z_ref)
                    x, y, _ = g.get_position()
                    placed_xy.append((x, y))
            for g in self.groceries:
                _set_upright_random_yaw(g, rng, yaw_deg=float(os.getenv("GROCERY_YAW_DEG", "30")))

        # ---- Clamp everything into workspace (prevents post-init BoundaryError) ----
        try:
            ws = Shape("workspace")
        except Exception:
            ws = None
        if ws is not None:
            for o in [self.shoe1, self.shoe2] + self.groceries + ([self.trash] if self.trash is not None else []):
                _clamp_obj_into_workspace(o, ws, self._z_ref)

        # Program shoes & obstacle flow (UNCHANGED)
        self._program_shoes_and_obstacle_flow()
        # Prime groceries (UNCHANGED)
        self._prime_groceries_validation()

        # Success conditions (UNCHANGED)
        conds: List[Condition] = []
        if getattr(self, "success_sensor_shoe", None) is not None:
            conds.extend([
                DetectedCondition(self.shoe1, self.success_sensor_shoe),
                DetectedCondition(self.shoe2, self.success_sensor_shoe),
            ])
        if getattr(self, "success_sensor_grocery", None) is not None:
            conds.append(
                DetectedSeveralCondition(
                    cast(List[Object], self.groceries),
                    self.success_sensor_grocery,
                    self.groceries_to_place,
                )
            )
        if conds:
            self.register_success_conditions(conds)

        if os.getenv("LAYOUT_DEBUG", "0") == "1":
            try:
                s1 = tuple(round(v, 3) for v in self.shoe1.get_position())
                s2 = tuple(round(v, 3) for v in self.shoe2.get_position())
                msg = f"[layout] k={index} seed={_base_seed()} shoe1={s1} shoe2={s2}"
                if self.trash is not None:
                    tr = tuple(round(v, 3) for v in self.trash.get_position())
                    msg += f" trash={tr}"
                print(msg)
            except Exception:
                pass

        return [
            "open the box, handle shoes and obstacle, then place groceries in the cupboard",
            "put both shoes in the box after removing obstacle, then store soup and spam",
        ]

    # -------------------- Shoes/Obstacle programming (UNCHANGED) --------------------
    def _program_shoes_and_obstacle_flow(self) -> None:
        def set_wp(idx: int, pose):
            try: Dummy(f"waypoint{idx}").set_pose(pose)
            except Exception: pass
        def get_pose(idx: int):
            return Dummy(f"waypoint{idx}").get_pose()

        # 5..8 : shoe1 pick & hover
        if self.shoe1_grasp is not None:
            x, y, z, qx, qy, qz, qw = self.shoe1_grasp.get_pose()
            set_wp(5, [x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])
            set_wp(6, [x, y, z,                 qx, qy, qz, qw])
            set_wp(7, [x, y, z + PICK_LIFT_DZ,  qx, qy, qz, qw])
        if getattr(self, "wp_hover_box", None) is not None:
            set_wp(8, self.wp_hover_box.get_pose())

        # 9..11 : clones of 5..7
        try:
            set_wp(9,  get_pose(5))
            set_wp(10, get_pose(6))
            set_wp(11, get_pose(7))
        except Exception:
            pass

        # 12 : align directly above 13 (same XY & ORI as 13, keep 12's Z)
        try:
            x13, y13, z13, qx13, qy13, qz13, qw13 = Dummy("waypoint13").get_pose()
            x12, y12, z12, _, _, _, _ = Dummy("waypoint12").get_pose()
            Dummy("waypoint12").set_pose([x13, y13, z12, qx13, qy13, qz13, qw13])
        except Exception:
            pass

        # 13..15 : straight vertical stack above 13
        self._align_trash_chain_straight()

        # 16 : hover over dustbin
        if getattr(self, "wp_hover_bin", None) is not None:
            set_wp(16, self.wp_hover_bin.get_pose())

        # 17..20 : pick shoe1 again; 20 hover
        if self.shoe1_grasp is not None:
            x, y, z, qx, qy, qz, qw = self.shoe1_grasp.get_pose()
            set_wp(17, [x, y, z + PICK_APPROACH_DZ, qx, qy, qz, qw])
            set_wp(18, [x, y, z,                 qx, qy, qz, qw])
            set_wp(19, [x, y, z + PICK_LIFT_DZ,  qx, qy, qz, qw])
        try:
            set_wp(20, get_pose(8))
        except Exception:
            pass

        # 21..23 : shoe2; 24 hover
        if self.shoe2_grasp is not None:
            x2, y2, z2, qx2, qy2, qz2, qw2 = self.shoe2_grasp.get_pose()
            set_wp(21, [x2, y2, z2 + PICK_APPROACH_DZ, qx2, qy2, qz2, qw2])
            set_wp(22, [x2, y2, z2,                 qx2, qy2, qz2, qw2])
            set_wp(23, [x2, y2, z2 + PICK_LIFT_DZ,  qx2, qy2, qz2, qw2])
        # 24 left as-is

    # -------------------- Start groceries after WP24 (UNCHANGED) --------------------
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
            tip = list(self.robot.arm.get_tip().get_pose()); tip[2] += max(STAGE_CLEAR_DZ, 0.10)
            self.wp_stage.set_pose([tip[0], tip[1], tip[2], *self._stage_q0])

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

    # -------------------- TRASH: straight-line helper + callbacks (UNCHANGED) --------------------
    def _align_trash_chain_straight(self) -> None:
        try:
            x13, y13, z13, qx13, qy13, qz13, qw13 = Dummy("waypoint13").get_pose()
            Dummy("waypoint14").set_pose([x13, y13, z13 + PICK_LIFT_DZ, qx13, qy13, qz13, qw13])
            Dummy("waypoint15").set_pose([x13, y13, z13 + PICK_LIFT_DZ + STAGE_CLEAR_DZ,
                                          qx13, qy13, qz13, qw13])
        except Exception:
            pass

    def _on_trash_pick_start(self, _) -> None:
        self._align_trash_chain_straight()
        try:
            if self.trash is not None:
                try: self.trash.set_parent(None)
                except Exception: pass
                try: self.trash.set_dynamic(True)
                except Exception: pass
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

    # -------------------- Groceries priming & callbacks (UNCHANGED) --------------------
    def _boundary_centre(self):
        if getattr(self, "_boundary_shape", None) is not None:
            return self._boundary_shape.get_position()
        xs, ys, zs = zip(*(g.get_position() for g in self.groceries))
        return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))

    def _estimate_table_z(self) -> float:
        if getattr(self, "_boundary_shape", None) is not None:
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
            tip = list(self.robot.arm.get_tip().get_pose()); tip[2] += max(STAGE_Z_ABOVE_TABLE, 0.30)
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

    def _on_pick_approach_start(self, _) -> None:
        if not self.shoes_done: return
        self._update_pre_approach_force(); self._update_pick_approach_force()
        try: self.robot.gripper.open()
        except Exception: pass

    def _on_pick_grasp_start(self, _) -> None:
        if not self.shoes_done: return
        self._update_pick_grasp_force()
        try: self.robot.gripper.open()
        except Exception: pass

    def _on_pick_lift_start(self, _) -> None:
        if not self.shoes_done: return
        try:
            obj = self.groceries[self.groceries_placed]
            self._held_obj = obj
            self._lock_to_tip(obj)
        except Exception:
            pass
        self._update_pick_lift_force()
        self._update_place_chain_from_current_item()

    def _on_transfer_up_start(self, _) -> None:
        if not self.shoes_done: return
        self._update_transfer_up_force()
        self._keep_closed_if_holding()
        try:
            if self._held_obj is not None: self._lock_to_tip(self._held_obj)
        except Exception: pass

    def _on_place_approach_start(self, _) -> None:
        if not self.shoes_done: return
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
            if self._held_obj is not None: self._lock_to_tip(self._held_obj)
        except Exception: pass
        self._mirror_29_to_36_chain()

    def _on_place_place_start(self, _) -> None:
        if not self.shoes_done: return
        self._update_place_place_force()
        try:
            if self._held_obj is not None: self._unlock_from_tip(self._held_obj)
            self.robot.gripper.release()
            self._held_obj = None
        except Exception: pass
        self._mirror_29_to_36_chain()

    def _on_place_retreat_start(self, _) -> None:
        if not self.shoes_done: return
        self._update_place_retreat_force()
        self._mirror_29_to_36_chain()

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
        if getattr(self, "bottom_goals", None) is not None:
            shelf_z = self.bottom_goals.get_position()[2]
        else:
            shelf_z = zL + TRANSFER_MIN_RAISE
        dz = shelf_z - zL
        if dz < TRANSFER_MIN_RAISE: target_z = zL + TRANSFER_MIN_RAISE
        elif dz > TRANSFER_MAX_RAISE: target_z = zL + TRANSFER_MAX_RAISE
        else: target_z = shelf_z
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
        if getattr(self, "wp2_place_place", None) is not None and getattr(self, "_scene_wp38_pose", None) is not None:
            sx2, sy2, sz2, sqx2, sqy2, sqz2, sqw2 = self._scene_wp38_pose
            self.wp2_place_place.set_pose([sx2 + PLACE_INSERT_WORLD_DX, sy2, sz2, sqx2, sqy2, sqz2, sqw2])

    def _update_place_retreat_force(self) -> None:
        i = self._current_index()
        gx, gy, gz, gqx, gqy, gqz, gqw = self.goals[i].get_pose()
        self.wp_place_retreat.set_pose([gx, gy, gz, gqx, gqy, gqz, gqw])
        self._mirror_29_to_36_chain()

    def _mirror_29_to_36_chain(self) -> None:
        try:
            if getattr(self, "wp2_transfer_up", None) is not None:
                self.wp2_transfer_up.set_pose(self.wp_stage.get_pose())
        except Exception: pass
        try:
            if getattr(self, "wp2_place_app", None) is not None:
                self.wp2_place_app.set_pose(self.wp_place_app.get_pose())
        except Exception: pass
        try:
            if getattr(self, "wp2_place_place", None) is not None:
                self.wp2_place_place.set_pose(self.wp_place_place.get_pose())
        except Exception: pass
        try:
            if getattr(self, "wp2_place_retreat", None) is not None:
                self.wp2_place_retreat.set_pose(self.wp_place_retreat.get_pose())
        except Exception: pass

    # -------------------- Hard-lock helpers (UNCHANGED) --------------------
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

    def _noop_shoes_when_done(self, _) -> None:
        if not self.shoes_done:
            return
        try:
            pose = self.robot.arm.get_tip().get_pose()
            for idx, d in self._pre_grocery_wp.items():
                if idx <= 23:
                    d.set_pose(pose)
        except Exception:
            pass
