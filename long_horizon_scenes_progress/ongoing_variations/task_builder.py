import os
import sys
from os.path import join, dirname, abspath, isfile
import time
import logging
from logging.handlers import RotatingFileHandler

CURRENT_DIR = dirname(abspath(__file__))

# ---------------- Logging setup ----------------
# Allow overrides via env; otherwise default to tools/logs/task_builder.log
LOG_DIR = os.getenv("TASK_BUILDER_LOG_DIR", join(CURRENT_DIR, 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.getenv("TASK_BUILDER_LOG_PATH", join(LOG_DIR, 'task_builder.log'))

logger = logging.getLogger('task_builder')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
logger.info('Task builder starting; logging to %s', LOG_PATH)

# Make sure we import the local RLBench checkout, not a site-installed copy.
sys.path.insert(0, join(CURRENT_DIR, '..'))

# ---- Tee stdout/stderr into the logger (without removing console output) ----
class _TeeToLogger:
    def __init__(self, logger_obj: logging.Logger, level: int, orig_stream):
        self._logger = logger_obj
        self._level = level
        self._orig = orig_stream
        self._buf = ''
        try:
            self.encoding = getattr(orig_stream, 'encoding', None)
        except Exception:
            self.encoding = None

    def write(self, message: str):
        try:
            self._orig.write(message)
        except Exception:
            pass
        if not isinstance(message, str):
            message = str(message)
        self._buf += message
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            try:
                self._logger.log(self._level, line)
            except Exception:
                pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        if self._buf:
            try:
                self._logger.log(self._level, self._buf)
            except Exception:
                pass
            self._buf = ''

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:
            return False

    def fileno(self):
        try:
            return self._orig.fileno()
        except Exception:
            return -1

    def __getattr__(self, name):
        return getattr(self._orig, name)

# Install the tees once
try:
    if not isinstance(sys.stdout, _TeeToLogger):
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeToLogger(logger, logging.INFO, _orig_stdout)
        sys.stderr = _TeeToLogger(logger, logging.ERROR, _orig_stderr)
except Exception:
    pass

# ---------------- Imports that might print (they'll be tee'd) ----------------
import traceback
import readline

from pyrep.const import RenderMode

from rlbench.backend import task
from rlbench.backend.const import TTT_FILE
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.object import Object
from pyrep.objects.joint import Joint
from pyrep.backend import sim
from rlbench.backend.scene import Scene
from rlbench.backend.exceptions import *
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.robot import Robot
from rlbench.utils import name_to_task_class
from task_validator import task_smoke, TaskValidationError
import shutil
import re

# Optional: Path class is not present in some PyRep builds; guard import.
try:
    from pyrep.objects.path import Path as PRPath  # type: ignore
except Exception:
    PRPath = None


def print_fail(message, end='\n'):
    """Keep red text in console, but still tee to log via stderr."""
    message = str(message)
    sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)


def setup_list_completer():
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    def list_completer(_, state):
        line = readline.get_line_buffer()
        if not line:
            return [c + " " for c in task_files][state]
        else:
            return [c + " " for c in task_files if c.startswith(line)][state]

    readline.parse_and_bind("tab: complete")
    readline.set_completer(list_completer)


class LoadedTask(object):

    def __init__(self, pr: PyRep, scene: Scene, robot: Robot):
        self.pr = pr
        self.scene = scene
        self.robot = robot
        self.task = self.task_class = self.task_file = None
        self._variation_index = 0

    def _load_task_to_scene(self):
        logger.info('Loading task into scene: class=%s, file=%s', str(self.task_class), self.task_file)
        self.scene.unload()
        self.task = self.task_class(
            self.pr, self.robot, self.task_file.replace('.py', ''))
        try:
            self.scene.load(self.task)
            logger.info('Task loaded into scene successfully.')
        except FileNotFoundError:
            # Fallback if TTM is missing; keep behavior identical to upstream.
            handle = Dummy.create()
            handle.set_name(self.task_file.replace('.py', ''))
            handle.set_model(True)
            try:
                self.task.get_base().set_position(Shape('workspace').get_position())
            except Exception:
                pass
            logger.warning('TTM not found; created placeholder base for %s', self.task_file)

    def _edit_new_task(self):
        task_file = input('What task would you like to edit?\n')
        task_file = task_file.strip(' ')
        if len(task_file) > 3 and task_file[-3:] != '.py':
            task_file += '.py'
        try:
            task_class = name_to_task_class(task_file)
        except Exception:
            print('There was no task named: %s. Would you like to create it?' % task_file)
            inp = input()
            if inp == 'y':
                self._create_python_file(task_file)
                task_class = name_to_task_class(task_file)
            else:
                print('Please pick a defined task in that case.')
                task_class, task_file = self._edit_new_task()
        return task_class, task_file

    def _create_python_file(self, task_file: str):
        with open(join(CURRENT_DIR, 'assets', 'task_template.txt'), 'r') as f:
            file_content = f.read()
        class_name = self._file_to_class_name(task_file)
        file_content = file_content % (class_name,)
        new_file_path = join(CURRENT_DIR, '../rlbench/tasks', task_file)
        if isfile(new_file_path):
            raise RuntimeError('File already exists. Will not override this.')
        with open(new_file_path, 'w+') as f:
            f.write(file_content)

    def _file_to_class_name(self, name):
        name = name.replace('.py', '')
        return ''.join([w[0].upper() + w[1:] for w in name.split('_')])

    def reload_python(self):
        try:
            task_class = name_to_task_class(self.task_file)
        except Exception:
            print_fail('The python file could not be loaded!')
            logger.exception('Failed to reload python for %s', self.task_file)
            traceback.print_exc()
            return None, None
        self.task = task_class(
            self.pr, self.robot, self.task_file.replace('.py', ''))
        self.scene.load(self.task)
        logger.info('Reloaded python for %s', self.task_file)

    def new_task(self):
        self._variation_index = 0
        self.task_class, self.task_file = self._edit_new_task()
        self._load_task_to_scene()
        self.pr.step_ui()
        print('You are now editing: %s' % str(self.task_class))

    def reset_variation(self):
        self._variation_index = 0

    def new_variation(self):
        try:
            self._variation_index += 1
            descriptions = self.scene.init_episode(
                self._variation_index % self.task.variation_count(),
                max_attempts=10)
            print('Task descriptions: ', descriptions)
            logger.info('New variation initialized: index=%d, descriptions=%s',
                        self._variation_index, descriptions)
        except (WaypointError, BoundaryError, Exception):
            logger.exception('Error during new variation')
            traceback.print_exc()
        self.pr.step_ui()

    def new_episode(self):
        try:
            descriptions = self.scene.init_episode(
                self._variation_index % self.task.variation_count(),
                max_attempts=10)
            print('Episode initialized successfully.')
            print('Task descriptions: ', descriptions)
            time.sleep(2.0)
            logger.info('Episode initialized: index=%d, descriptions=%s',
                        self._variation_index, descriptions)
        except (WaypointError, BoundaryError, Exception) as ex:
            logger.exception('Error during new episode')
            traceback.print_exc()
            # Convenience: if the thrown message contains "waypoint N", print basic info.
            try:
                msg = str(ex)
                m = re.search(r"waypoint\s*(\d+)", msg)
                if m is not None:
                    idx = int(m.group(1))
                    print(f"[Debug] Gathering scene details for waypoint index {idx}...")
                    self._debug_waypoint_index(idx)
            except Exception:
                pass
            self.scene.reset()
        self.pr.step_ui()

    def _debug_waypoint_index(self, idx: int):
        """List objects named waypoint<idx> under the task base (name, parent, type)."""
        try:
            base = self.task.get_base()
            all_objs = base.get_objects_in_tree(exclude_base=True)
            matches = []
            for o in all_objs:
                try:
                    name = o.get_name()
                except Exception:
                    continue
                m = re.fullmatch(r"waypoint(\d+)", name)
                if not m:
                    continue
                try:
                    num = int(m.group(1))
                except Exception:
                    continue
                if num == idx:
                    parent_name = ''
                    try:
                        parent = o.get_parent()
                        parent_name = parent.get_name() if parent is not None else ''
                    except Exception:
                        parent_name = ''
                    type_name = o.__class__.__name__
                    if PRPath is not None:
                        try:
                            _ = PRPath(o.get_handle())
                            type_name = 'Path'
                        except Exception:
                            pass
                    matches.append((name, parent_name, type_name))
            if not matches:
                print(f"[Debug] No objects named waypoint{idx} found under the task base.")
                return
            print(f"[Debug] Objects with index {idx} (name, parent, type):")
            for (n, p, t) in matches:
                print(f"  - {n}  parent={p}  type={t}")
            if len(matches) > 1:
                print(f"[Debug] Multiple waypoint{idx} objects found. RLBench will pick by numeric order but duplicates can cause unintended ordering.")
        except Exception:
            pass

    def new_demo(self):
        try:
            self.scene.get_demo(False, randomly_place=True)
        except (WaypointError, NoWaypointsError, DemoError, Exception):
            logger.exception('Error during demo')
            traceback.print_exc()
        success, _ = self.task.success()
        if success:
            print("Demo was a success!")
            time.sleep(1.5)
            logger.info('Demo success')
        else:
            logger.info('Demo finished without success')
        self.scene.reset()
        self.pr.step_ui()
        self.pr.step_ui()

    def save_task(self):
        base_name = self.task_file.replace('.py', '')
        ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms', base_name + '.ttm')
        try:
            base = self.task.get_base()
        except Exception:
            try:
                base = Dummy.create()
                base.set_name(base_name)
                base.set_model(True)
                try:
                    ws = Shape('workspace')
                    base.set_position(ws.get_position())
                except Exception:
                    pass
                try:
                    br = Dummy('boundary_root')
                    br.set_parent(base)
                except Exception:
                    pass
                print(f"[Auto-Fix] Created base model '{base_name}' and attached boundary_root if present.")
            except Exception:
                print_fail(
                    f"Cannot save .ttm: base model '{base_name}' not found and auto-create failed.\n"
                    "In CoppeliaSim: create a Dummy named exactly the task name, set it as a Model,\n"
                    "and parent all task objects under it. Then press 's' again.")
                return
        try:
            base.save_model(ttm_path)
            print('Task saved to:', ttm_path)
        except Exception as e:
            print_fail(f"Failed to save .ttm to {ttm_path}: {e}")

    def run_task_validator(self):
        if not ProximitySensor.exists('success'):
            print_fail(
                "No 'success' proximity sensor found in the scene. "
                "Did the TTM file for the task load correctly?")
            return

        print('About to perform task validation.')
        print("What variation to test? Pick int in range: 0 to %d, or -1 to "
              "test all. Or press 'e' to exit."
              % self.task.variation_count())
        inp = input()
        if inp == 'e':
            logger.info('Validator cancelled by user')
            return
        self.pr.start()
        try:
            v = int(inp)
            v = v if v < 0 else v % self.task.variation_count()
            logger.info('Validator starting: variation=%d', v)
            task_smoke(self.task, self.scene, variation=v)
            logger.info('Validator finished')
        except TaskValidationError:
            logger.exception('Validator failed')
            traceback.print_exc()
        self.pr.stop()

    def rename(self):
        print('Enter new name (or q to abort).')
        inp = input()
        if inp == 'q':
            logger.info('Rename cancelled')
            return

        name = inp.replace('.py', '')
        python_file = name + '.py'

        handle = Dummy(self.task_file.replace('.py', ''))
        handle.set_name(name)

        old_file_path = join(CURRENT_DIR, '../rlbench/tasks', self.task_file)
        old_class_name = self._file_to_class_name(self.task_file)
        new_class_name = self._file_to_class_name(name)
        with open(old_file_path, 'r') as f:
            content = f.read()
        content = content.replace(old_class_name, new_class_name)
        with open(old_file_path, 'w') as f:
            f.write(content)

        new_file_path = join(CURRENT_DIR, '../rlbench/tasks', python_file)
        os.rename(old_file_path, new_file_path)

        old_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                            self.task_file.replace('.py', '.ttm'))
        new_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                            python_file.replace('.py', '.ttm'))
        os.rename(old_ttm_path, new_ttm_path)

        self.task_file = python_file
        self.reload_python()
        self.save_task()
        print('Rename complete!')

    def duplicate_task(self):
        print('Enter new name for duplicate (or q to abort).')
        inp = input()
        if inp == 'q':
            logger.info('Duplicate cancelled')
            return

        name = inp.replace('.py', '')
        new_python_file = name + '.py'

        old_file_path = join(CURRENT_DIR, '../rlbench/tasks', self.task_file)
        old_class_name = self._file_to_class_name(self.task_file)
        new_file_path = join(CURRENT_DIR, '../rlbench/tasks', new_python_file)
        new_class_name = self._file_to_class_name(name)

        if os.path.isfile(new_file_path):
            print('File: %s already exists!' % new_file_path)
            return

        handle = Dummy(self.task_file.replace('.py', ''))
        handle.set_name(name)

        with open(old_file_path, 'r') as f:
            content = f.read()
        content = content.replace(old_class_name, new_class_name)
        with open(new_file_path, 'w+') as f:
            f.write(content)

        old_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                            self.task_file.replace('.py', '.ttm'))
        new_ttm_path = join(CURRENT_DIR, '../rlbench/task_ttms',
                            new_python_file.replace('.py', '.ttm'))
        shutil.copy(old_ttm_path, new_ttm_path)

        self.task_file = new_python_file
        self.reload_python()
        self.save_task()
        print('Duplicate complete!')


def main():
    setup_list_completer()

    task_class, task_file = _prompt_for_task()
    logger.info('User selected task: %s', task_file)

    pr, ttt_to_launch, task_ttt_path = launch_pyrep_with_task(task_file)
    print(f"[Scene] Loaded: {ttt_to_launch}")

    # --- Try to discover an existing Panda or its #n variants and normalize names ---
    def _list_candidates(prefix: str, max_suffix: int = 20):
        names = []
        try:
            sim.simGetObjectHandle(prefix)
            names.append(prefix)
        except Exception:
            pass
        for i in range(max_suffix + 1):
            cand = f"{prefix}#{i}"
            try:
                sim.simGetObjectHandle(cand)
                names.append(cand)
            except Exception:
                continue
        return names

    panda_candidates = _list_candidates('Panda')
    if panda_candidates:
        print(f"[Debug] Found Panda candidates: {panda_candidates}")
    else:
        print("[Debug] No Panda or Panda#<n> found in scene yet.")

    chosen_count = 0
    chosen_suffix = ''
    for c in range(0, 6):
        sfx = '' if c == 0 else f'#{c-1}'
        ik_name = 'Panda_ik' if c == 0 else f'Panda_ik#{c-1}'
        coll_name = 'Panda_arm' if c == 0 else f'Panda_arm#{c-1}'
        ik_ok = coll_ok = False
        try:
            sim.simGetIkGroupHandle(ik_name)
            ik_ok = True
        except Exception:
            pass
        try:
            sim.simGetCollectionHandle(coll_name)
            coll_ok = True
        except Exception:
            pass
        if ik_ok or coll_ok:
            chosen_count = c
            chosen_suffix = sfx
            break
    print(f"[Debug] Using suffix '{chosen_suffix}' (count={chosen_count}) based on IK/collection")

    def _exists(name: str) -> bool:
        try:
            sim.simGetObjectHandle(name)
            return True
        except Exception:
            return False

    def _rename(src: str, dst: str) -> bool:
        if not _exists(src):
            return False
        if _exists(dst):
            return True
        try:
            Object(sim.simGetObjectHandle(src)).set_name(dst)
            logger.info("Renamed '%s' -> '%s'", src, dst)
            return True
        except Exception:
            return False

    names_core = ['Panda', 'Panda_gripper', 'Panda_target', 'Panda_tip']
    joint_names = [f'Panda_joint{i}' for i in range(1, 8)]
    if chosen_suffix == '':
        for base in names_core + joint_names:
            if _exists(base):
                continue
            for k in range(0, 6):
                _rename(f"{base}#{k}", base)
    else:
        for base in names_core + joint_names:
            suff = base + chosen_suffix
            if _exists(suff):
                continue
            if _exists(base):
                _rename(base, suff)
            else:
                for k in range(0, 6):
                    if k == int(chosen_suffix[1:]):
                        continue
                    if _rename(f"{base}#{k}", suff):
                        break

    last_err = None
    robot = None
    try:
        robot = Robot(Panda(count=chosen_count), PandaGripper(count=chosen_count))
        print(f"[Debug] Using Panda with count={chosen_count}")
    except Exception as e:
        last_err = e
    if robot is None:
        need = ['Panda', 'Panda_target', 'Panda_tip'] + [f'Panda_joint{i}' for i in range(1, 8)]
        missing = []
        for nm in need:
            found = False
            try:
                sim.simGetObjectHandle(nm)
                found = True
            except Exception:
                for k in range(0, 6):
                    try:
                        sim.simGetObjectHandle(f"{nm}#{k}")
                        found = True
                        break
                    except Exception:
                        continue
            if not found:
                missing.append(nm)
        ik_states = []
        for c in range(0, 6):
            nm = 'Panda_ik' if c == 0 else f'Panda_ik#{c-1}'
            try:
                sim.simGetIkGroupHandle(nm)
                ik_states.append((nm, True))
            except Exception:
                ik_states.append((nm, False))
        coll_states = []
        for c in range(0, 6):
            nm = 'Panda_arm' if c == 0 else f'Panda_arm#{c-1}'
            try:
                sim.simGetCollectionHandle(nm)
                coll_states.append((nm, True))
            except Exception:
                coll_states.append((nm, False))
        print_fail(f"Failed to create Robot with Panda across counts 0-5. Missing objects: {missing}. IK groups: {ik_states}. Collections: {coll_states}. Last error: {last_err}")
        try:
            pr.stop()
        except Exception:
            pass
        try:
            pr.shutdown()
        except Exception:
            pass
        sys.exit(1)

    cam_config = CameraConfig(rgb=True, depth=False, mask=False, render_mode=RenderMode.OPENGL)
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.right_shoulder_camera = cam_config
    obs_config.left_shoulder_camera = cam_config
    obs_config.overhead_camera = cam_config
    obs_config.wrist_camera = cam_config
    obs_config.front_camera = cam_config

    scene = Scene(pr, robot, obs_config)
    loaded_task = LoadedTask(pr, scene, robot)

    loaded_task.task_class, loaded_task.task_file = task_class, task_file
    loaded_task._load_task_to_scene()
    pr.step_ui()

    print('  ,')
    print(' /(  ___________')
    print('|  >:===========`  Welcome to task builder!')
    print(' )(')
    print(' ""')

    print('You are now editing: %s' % str(task_class))
    print(f"Scene file loaded: {ttt_to_launch}")

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n-----------------\n')
        print('The python file will be reloaded when simulation is restarted.')
        print('(q) to quit.')
        if pr.running:
            print(' (+) stop the simulator')
            print(' (v) for task variation.')
            print(' (e) for episode of same variation.')
            print(' (d) for demo.')
            print(' (p) for running the sim for 100 steps (with rendering).')
        else:
            print(' (!) to run task validator.')
            print(' (+) run the simulator')
            print(' (n) for new task.')
            print(' (s) to save the .ttm')
            print(' (S) to export the scene (.ttt)')
            print(' (r) to rename the task')
            print(' (u) to duplicate/copy the task')
        print(f"[Info] Current scene path: {ttt_to_launch}")

        inp = input()
        logger.info('Key pressed: %s', inp)

        if inp == 'q':
            logger.info('Quitting task builder')
            break

        if pr.running:
            if inp == '+':
                pr.stop()
                pr.step_ui()
                logger.info('Simulator stopped')
            elif inp == 'p':
                [(pr.step(), scene.get_observation()) for _ in range(100)]
                logger.info('Simulated 100 steps with rendering')
            elif inp == 'd':
                loaded_task.new_demo()
            elif inp == 'v':
                loaded_task.new_variation()
            elif inp == 'e':
                loaded_task.new_episode()
        else:
            if inp == '+':
                loaded_task.reload_python()
                loaded_task.reset_variation()
                pr.start()
                pr.step_ui()
                logger.info('Simulator started')
            elif inp == 'n':
                inp2 = input('Do you want to save the current task first?\n')
                if inp2 == 'y':
                    loaded_task.save_task()
                    logger.info('Saved before starting new task')

                pr.stop()
                pr.shutdown()

                task_class, task_file = _prompt_for_task()
                pr, ttt_to_launch, task_ttt_path = launch_pyrep_with_task(task_file)

                robot = None
                for c in range(0, 6):
                    try:
                        robot = Robot(Panda(count=c), PandaGripper(count=c))
                        print(f"[Debug] Using Panda with count={c}")
                        break
                    except Exception:
                        continue
                if robot is None:
                    print_fail('Could not create Panda robot in the newly loaded scene. Please ensure Panda, joints, target/tip, IK group and arm collection exist (possibly with #0 suffix).')
                    sys.exit(1)
                scene = Scene(pr, robot, obs_config)

                loaded_task.pr = pr
                loaded_task.scene = scene
                loaded_task.robot = robot
                loaded_task.task_class, loaded_task.task_file = task_class, task_file
                loaded_task._load_task_to_scene()
                pr.step_ui()
                print('You are now editing: %s' % str(task_class))

            elif inp == 's':
                loaded_task.save_task()
            elif inp == '!':
                loaded_task.run_task_validator()
            elif inp == 'S':
                try:
                    current_task_file = loaded_task.task_file if loaded_task.task_file else task_file
                    chosen_name = current_task_file.replace('.py','')
                    export_path = join(CURRENT_DIR, '..', 'rlbench', 'task_ttms', f"{chosen_name}.ttt")
                    pr.export_scene(export_path)
                    print(f'Scene exported to: {export_path}')
                    logger.info('Scene exported to %s', export_path)
                except Exception:
                    logger.exception('Failed exporting scene')
            elif inp == 'r':
                loaded_task.rename()
            elif inp == 'u':
                loaded_task.duplicate_task()

    pr.stop()
    pr.shutdown()
    print('Done. Goodbye!')
    logger.info('Task builder shutdown')


def _prompt_for_task() -> tuple:
    task_file = input('What task would you like to edit?\n')
    task_file = task_file.strip(' ')
    if len(task_file) > 3 and task_file[-3:] != '.py':
        task_file += '.py'
    try:
        task_class = name_to_task_class(task_file)
    except Exception:
        print('There was no task named: %s. Would you like to create it?' % task_file)
        inp = input()
        if inp == 'y':
            with open(join(CURRENT_DIR, 'assets', 'task_template.txt'), 'r') as f:
                file_content = f.read()
            class_name = ''.join([w[0].upper() + w[1:] for w in task_file.replace('.py','').split('_')])
            file_content = file_content % (class_name,)
            new_file_path = join(CURRENT_DIR, '../rlbench/tasks', task_file)
            if isfile(new_file_path):
                raise RuntimeError('File already exists. Will not override this.')
            with open(new_file_path, 'w+') as f:
                f.write(file_content)
            task_class = name_to_task_class(task_file)
        else:
            return _prompt_for_task()
    return task_class, task_file


def launch_pyrep_with_task(task_file: str):
    """Launch a scene for the chosen task.

    Behavior:
      - If rlbench/task_ttms/<task>.ttt exists, load that per-task scene.
      - Otherwise, fall back to rlbench/task_design.ttt (stock template).
      - Export (S) always writes back to rlbench/task_ttms/<task>.ttt.
    """
    base_rlbench_dir = join(CURRENT_DIR, '..', 'rlbench')
    task_ttms_dir = join(base_rlbench_dir, 'task_ttms')
    os.makedirs(task_ttms_dir, exist_ok=True)

    chosen_name = task_file.replace('.py', '')
    stock_scene = join(base_rlbench_dir, 'task_design.ttt')
    task_specific_ttt_path = join(task_ttms_dir, f"{chosen_name}.ttt")

    ttt_to_launch = task_specific_ttt_path if os.path.isfile(task_specific_ttt_path) else stock_scene

    pr = PyRep()
    pr.launch(ttt_to_launch, responsive_ui=True)
    pr.step_ui()
    return pr, ttt_to_launch, task_specific_ttt_path


if __name__ == '__main__':
    main()
