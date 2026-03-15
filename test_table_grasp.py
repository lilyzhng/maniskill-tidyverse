"""Table-top grasp test: table + red block, multi-angle pick with diagnostics."""
import sys
import os
import argparse
import numpy as np
import torch
import sapien
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
import tidyverse_agent  # noqa: F401 — registers 'tidyverse' robot
import mani_skill.envs  # noqa: F401 — registers ManiSkill envs

from mplib import Pose as MPPose
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.collision_detection.fcl import (
    Box, Capsule, Convex, Sphere, BVHModel, Halfspace, Cylinder,
    CollisionObject, FCLObject,
)
from sapien.physx import (
    PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh, PhysxCollisionShapeSphere,
    PhysxCollisionShapeTriangleMesh, PhysxCollisionShapePlane,
    PhysxCollisionShapeCylinder, PhysxArticulationLinkComponent,
)
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
import mplib.sapien_utils.conversion as _conv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.sensors.camera import CameraConfig

# ─── Constants ────────────────────────────────────────────────────────────────

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81   # Robotiq 85 joint range [0, 0.81] rad
PRE_GRASP_HEIGHT = 0.08  # metres above grasp target
LIFT_HEIGHT = 0.15        # metres above grasp target

# Planning masks: True = locked joint, False = free joint
# Layout: [base_x, base_y, base_yaw, arm×7, gripper×6]
MASK_ARM_ONLY = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)


# ─── Monkey-patch: apply scale to Robotiq convex collision meshes ─────────────

@staticmethod
def _convert_physx_component(comp):
    shapes, shape_poses = [], []
    for shape in comp.collision_shapes:
        shape_poses.append(MPPose(shape.local_pose))
        if isinstance(shape, PhysxCollisionShapeBox):
            geom = Box(side=shape.half_size * 2)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            geom = Capsule(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            verts = shape.vertices
            if not np.allclose(shape.scale, 1.0):
                verts = verts * np.array(shape.scale)
            geom = Convex(vertices=verts, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeSphere):
            geom = Sphere(radius=shape.radius)
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            geom = BVHModel()
            geom.begin_model()
            geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)
            geom.end_model()
        elif isinstance(shape, PhysxCollisionShapePlane):
            n = shape_poses[-1].to_transformation_matrix()[:3, 0]
            d = n.dot(shape_poses[-1].p)
            geom = Halfspace(n=n, d=d)
            shape_poses[-1] = MPPose()
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            geom = Cylinder(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        else:
            continue
        shapes.append(CollisionObject(geom))
    if not shapes:
        return None
    name = (comp.name if isinstance(comp, PhysxArticulationLinkComponent)
            else _conv.convert_object_name(comp.entity))
    return FCLObject(name, comp.entity.pose, shapes, shape_poses)

SapienPlanningWorld.convert_physx_component = _convert_physx_component


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_action(arm_qpos, gripper, base_cmd):
    """Build a single action tensor: [arm(7), gripper(1), base(3)]."""
    act = np.concatenate([arm_qpos, [gripper], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)


def wait_until_stable(step_fn, hold, robot, max_steps=300,
                      vel_thresh=1e-3, window=10):
    """Step simulation until robot velocities settle."""
    stable_count = 0
    for si in range(max_steps):
        step_fn(hold)
        qvel = robot.get_qvel().cpu().numpy()[0]
        if np.max(np.abs(qvel)) < vel_thresh:
            stable_count += 1
            if stable_count >= window:
                print(f"    Stabilized after {si + 1} steps")
                return si + 1
        else:
            stable_count = 0
    print(f"    WARNING: not stable after {max_steps} steps "
          f"(max |qvel|={np.max(np.abs(qvel)):.4f})")
    return max_steps


def execute_trajectory(traj, step_fn, gripper, base_cmd):
    """Execute a planned trajectory, sending arm joint targets each step."""
    for i in range(traj.shape[0]):
        step_fn(make_action(traj[i, 3:10], gripper, base_cmd))


def plan_and_move(label, planner, pw, pose, qpos, mask, step_fn,
                  gripper, base_cmd, planning_time=5.0):
    """Plan to a pose and execute. Returns True on success."""
    try:
        planner.update_from_simulation()
    except Exception:
        pass
    cq = qpos() if callable(qpos) else qpos
    result = planner.plan_pose(pose, cq, mask=mask, planning_time=planning_time)
    if result['status'] == 'Success':
        n = result['position'].shape[0]
        print(f"  {label}: OK  ({n} waypoints, {result['duration']:.2f}s)")
        execute_trajectory(result['position'], step_fn, gripper, base_cmd)
        return True
    diagnose_failure(label, result, pose, cq, planner, pw, mask)
    return False


def diagnose_failure(label, result, target_pose, current_qpos,
                     planner, pw, mask=None):
    """Print detailed diagnostics when planning fails."""
    print(f"  {label}: FAILED — {result['status']}")
    print(f"    Target pos:  {np.array(target_pose.p)}")
    print(f"    Target quat: {np.array(target_pose.q)}")
    print(f"    Current qpos (arm): {current_qpos[3:10]}")

    # Obstacles
    obj_names = pw.get_object_names()
    print(f"    Obstacles ({len(obj_names)}):")
    for oname in obj_names:
        obj = pw.get_object(oname)
        print(f"      - {oname}  pose={obj.pose}")

    # Current-state collisions
    collisions = pw.check_collision()
    if collisions:
        print(f"    Current-state collisions ({len(collisions)}):")
        for c in collisions:
            print(f"      - {c.link_name1}({c.object_name1}) "
                  f"<-> {c.link_name2}({c.object_name2})")

    # Standalone IK check
    try:
        ik_status, ik_solutions = planner.IK(
            target_pose, current_qpos, mask=mask, n_init_qpos=40, verbose=True)
        if ik_solutions is not None:
            print(f"    IK check: {len(ik_solutions)} solution(s) "
                  f"— RRT failed to find collision-free path")
            for i, q in enumerate(ik_solutions):
                print(f"      solution {i}: arm_qpos={q[3:10]}")
        else:
            print(f"    IK check: {ik_status}")
            # Retry without mask to isolate the cause
            ik2, sols2 = planner.IK(
                target_pose, current_qpos, mask=None,
                n_init_qpos=40, verbose=True)
            if sols2 is not None:
                print(f"    IK (no mask): {len(sols2)} solution(s) "
                      f"— mask is too restrictive")
            else:
                print(f"    IK (no mask): {ik2} — pose is unreachable")
    except Exception as e:
        print(f"    IK check error: {e}")


def build_grasp_poses(block_pos, arm_base):
    """Compute grasp target poses relative to the block."""
    yaw = np.arctan2(block_pos[1] - arm_base[1], block_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return [
        ('Top-Down',
         block_pos + [0, 0, 0],
         [0, 1, 0, 0]),
        ('Front',
         block_pos.copy(),
         list(R.from_euler('yz', [np.pi / 2, yaw]).as_quat()[[3, 0, 1, 2]])),
        ('Angled45',
         block_pos + [-0.02 * cos_y, -0.02 * sin_y, 0.02],
         list(euler2quat(0, 3 * np.pi / 4, yaw))),
    ]


# ─── Camera setup ────────────────────────────────────────────────────────────

def setup_camera():
    """Override default camera to a raised, zoomed-out viewpoint."""
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.utils import sapien_utils as ms_sapien_utils

    @property
    def _cam(self):
        pose = ms_sapien_utils.look_at(
            eye=[0.9, 1.0, 1.5], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    PickCubeEnv._default_human_render_camera_configs = _cam


# ─── Scene setup ──────────────────────────────────────────────────────────────

def create_table(scene, x, height):
    tb = scene.create_actor_builder()
    half = [0.3, 0.3, height / 2]
    tb.add_box_collision(half_size=half)
    tb.add_box_visual(half_size=half,
                      material=sapien.render.RenderMaterial(
                          base_color=[0.6, 0.4, 0.2, 1.0]))
    table = tb.build_static(name="table")
    table.set_pose(sapien.Pose(p=[x, 0, height / 2]))
    return table


def create_block(scene, x, table_height):
    bb = scene.create_actor_builder()
    half = [0.02, 0.02, 0.02]
    bb.add_box_collision(half_size=half)
    bb.add_box_visual(half_size=half,
                      material=sapien.render.RenderMaterial(
                          base_color=[1.0, 0.2, 0.2, 1.0]))
    block = bb.build(name="red_block")
    block.set_pose(sapien.Pose(p=[x, 0, table_height + 0.04]))
    return block


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Table-top grasp test")
    parser.add_argument('--render', default='human',
                        choices=['human', 'rgb_array'],
                        help='human=GUI window, rgb_array=save video')
    parser.add_argument('--robot-x', type=float, default=-0.3)
    parser.add_argument('--table-x', type=float, default=0.0)
    parser.add_argument('--table-height', type=float, default=0.762,
                        help='Table height in metres (default: 30 in)')
    args = parser.parse_args()

    # --- Environment ---
    setup_camera()
    env = gym.make('PickCube-v1', num_envs=1, robot_uids='tidyverse',
                   control_mode='whole_body', render_mode=args.render)
    video_dir = os.path.join(os.path.dirname(__file__), 'videos')
    if args.render == 'rgb_array':
        env = RecordEpisode(env, output_dir=video_dir, save_video=True,
                            max_steps_per_video=10000, video_fps=30)
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    scene_ms = env.unwrapped.scene
    is_human = (args.render == 'human')

    def step_fn(action):
        env.step(action)
        if is_human:
            env.render()

    # --- Scene objects ---
    robot.set_pose(sapien.Pose(p=[args.robot_x, 0, 0]))
    create_table(scene_ms, args.table_x, args.table_height)
    block = create_block(scene_ms, args.table_x, args.table_height)

    # --- Stabilize ---
    base_cmd = np.array([args.robot_x, 0.0, 0.0])
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)

    print("Waiting for robot to stabilize...")
    wait_until_stable(step_fn, hold, robot)

    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    block_pos = block.pose.p[0].cpu().numpy()
    print(f"Arm base: {arm_base}")
    print(f"Block:    {block_pos}  (dist XY: {np.linalg.norm(arm_base[:2] - block_pos[:2]):.3f}m)")

    # --- Planner ---
    pw = SapienPlanningWorld(scene, [robot._objs[0]])
    eef = next(n for n in pw.get_planned_articulations()[0]
               .get_pinocchio_model().get_link_names() if 'eef' in n)
    planner = SapienPlanner(pw, move_group=eef)

    # --- Grasp loop ---
    grasps = build_grasp_poses(block_pos, arm_base)
    get_qpos = lambda: robot.get_qpos().cpu().numpy()[0]

    for gi, (name, target_p, target_q) in enumerate(grasps):
        print(f"\n{'='*50}")
        print(f"[{gi + 1}/{len(grasps)}] {name}")
        print(f"  Target: pos={target_p}  quat={target_q}")

        # Reset arm to home
        qpos = get_qpos()
        qpos[3:10] = ARM_HOME
        qpos[10:] = 0.0
        robot.set_qpos(torch.tensor(qpos, dtype=torch.float32).unsqueeze(0))
        print("  Stabilizing arm reset...")
        wait_until_stable(step_fn, hold, robot, max_steps=200)
        print(f"  Base: {get_qpos()[:3]}  Arm: {get_qpos()[3:10]}")

        target_q_arr = np.array(target_q)

        # 1. Pre-grasp
        pre_pose = MPPose(p=np.array(target_p) + [0, 0, PRE_GRASP_HEIGHT],
                          q=target_q_arr)
        solved = False
        for mode, mask in [("arm-only", MASK_ARM_ONLY),
                           ("whole-body", MASK_WHOLE_BODY)]:
            if plan_and_move(f"Pre-grasp ({mode})", planner, pw, pre_pose,
                             get_qpos, mask, step_fn, GRIPPER_OPEN, base_cmd):
                used_mask = mask
                solved = True
                break
        if not solved:
            print("  SKIPPED — no solution")
            continue

        # 2. Approach
        approach_pose = MPPose(p=np.array(target_p), q=target_q_arr)
        plan_and_move("Approach", planner, pw, approach_pose,
                      get_qpos, used_mask, step_fn, GRIPPER_OPEN, base_cmd)

        # 3. Close gripper
        aq = get_qpos()[3:10]
        for _ in range(30):
            step_fn(make_action(aq, GRIPPER_CLOSED, base_cmd))

        # 4. Lift
        lift_pose = MPPose(p=np.array(target_p) + [0, 0, LIFT_HEIGHT],
                           q=target_q_arr)
        plan_and_move("Lift", planner, pw, lift_pose,
                      get_qpos, used_mask, step_fn, GRIPPER_CLOSED, base_cmd,
                      planning_time=3.0)

        # Hold for observation
        last_act = make_action(get_qpos()[3:10], GRIPPER_CLOSED, base_cmd)
        for _ in range(60):
            step_fn(last_act)

    # --- Finish ---
    if is_human:
        print("\nDone! Close the window to exit.")
        while True:
            env.step(hold)
            env.render()
    else:
        env.close()
        print(f"\nDone! Video saved to {video_dir}/")


if __name__ == '__main__':
    main()
