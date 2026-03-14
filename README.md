# ManiSkill TidyVerse Robot

TidyVerse robot agent for [ManiSkill3](https://github.com/haosulab/ManiSkill) — a Franka Panda arm on a TidyBot mobile base with a Robotiq 85 gripper. Matches the real [TidyBot](https://tidybot.cs.princeton.edu/) hardware.

## Robot Specs

- **Arm:** Franka Panda 7-DOF
- **Gripper:** Robotiq 2F-85 (parallel jaw)
- **Base:** 3-DOF mobile base (x, y, yaw)
- **Total active joints:** 16 (3 base + 7 arm + 6 gripper)
- **EE link:** `eef`

## Install

```bash
pip install mani_skill==3.0.0b22 mplib==0.2.1 pycollada
```

## Setup

```bash
git clone https://github.com/shaoyifei96/maniskill-tidyverse
cd maniskill-tidyverse

# Create symlinks (required — URDF mesh paths are relative)
ln -sf $(python3 -c "import mani_skill; print(mani_skill.__path__[0])")/assets/robots/panda/franka_description franka_description
ln -sf ~/.maniskill/data/robots/robotiq_2f/meshes robotiq_meshes
```

## Quick Start

```python
import sys
sys.path.insert(0, '/path/to/maniskill-tidyverse')
import tidyverse_agent  # registers 'tidyverse' robot via @register_agent()
import mani_skill.envs

import gymnasium as gym

# Headless
env = gym.make('RoboCasaKitchen-v1', num_envs=1,
               robot_uids='tidyverse',
               control_mode='pd_ee_delta_pose')
obs, info = env.reset(seed=42)

# With GUI (requires DISPLAY)
env = gym.make('RoboCasaKitchen-v1', render_mode='human', num_envs=1,
               robot_uids='tidyverse',
               control_mode='pd_ee_delta_pose')
```

Works with any ManiSkill3 environment (`PickCube-v1`, `RoboCasaKitchen-v1`, etc.).

## Control Modes

| Mode | DOF | Action Format |
|------|-----|---------------|
| `pd_ee_delta_pose` | 10 | `[dx,dy,dz, dax,day,daz, gripper, base_vx, base_vy, base_vyaw]` |
| `pd_ee_pose` | 10 | `[x,y,z, ax,ay,az, gripper, base_vx, base_vy, base_vyaw]` |
| `pd_joint_pos` | 16 | `[base_x, base_y, base_yaw, j1-j7, 6 gripper joints]` |
| `pd_joint_delta_pos` | 16 | `[Δbase_x, Δbase_y, Δbase_yaw, Δj1-Δj7, 6 gripper joints]` |

## RoboCasa Kitchens

120 kitchen configurations (10 layouts × 12 styles) are available via `RoboCasaKitchen-v1`. Change kitchens by varying the seed:

```python
obs, info = env.reset(seed=0)   # kitchen style 0
obs, info = env.reset(seed=42)  # different kitchen
```

### Object Spawning

Place objects on counter surfaces using fixture data:

```python
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType, fixture_is_type

unwrapped = env.unwrapped
sb = unwrapped.scene_builders['scene_0']
fixtures = sb.scene_data[0]['fixtures']

# Find counters
counters = {k: v for k, v in fixtures.items() if fixture_is_type(v, FixtureType.COUNTER)}

# Get surface position
counter = counters['counter_main_main_group']
surface_z = counter.pos[2] + counter.size[2] / 2

# Spawn a cube on the counter
builder = unwrapped.scene.create_actor_builder()
builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
builder.add_box_visual(half_size=[0.02, 0.02, 0.02], material=sapien.render.RenderMaterial(base_color=[1,0,0,1]))
cube = builder.build(name='red_cube')
cube.set_pose(sapien.Pose(p=[counter.pos[0], counter.pos[1], surface_z + 0.02]))
```

153 object categories available (fruits, vegetables, kitchenware, utensils, packaged food, etc.).

## Motion Planning (mplib)

mplib 0.2.1 is supported. Note: requires significant RAM (~16GB+) when combined with ManiSkill's SAPIEN renderer.

```python
import mplib

link_names = [l.get_name() for l in robot.get_links()]
joint_names = [j.get_name() for j in robot.get_active_joints()]

planner = mplib.Planner(
    urdf=agent.urdf_path,
    user_link_names=link_names,
    user_joint_names=joint_names,
    move_group="eef",
)

# Set robot base pose
bp = robot.pose.p[0].cpu().numpy()
bq = robot.pose.q[0].cpu().numpy()
planner.set_base_pose(np.concatenate([bp, bq]))

# Plan to target pose
result = planner.plan_pose(target_pose, current_qpos, time_step=env.unwrapped.control_timestep)
```

For lower memory usage, the `pd_ee_delta_pose` controller handles IK internally without mplib.

## Known Limitations

- **`RoboCasaKitchen-v1` is a scene viewer only** — `evaluate()` returns `{}`, no task definitions or success conditions
- **No fixture interaction** — `is_open()`/`set_door_state()` are stubs in ManiSkill3's RoboCasa port
- **Robot placement** — spawns at origin; no task logic positions it near fixtures
- **ManiSkill warns** `"tidyverse is not in the task's list of supported robots"` — safe to ignore
- **mplib + RoboCasa OOM** — on machines with <32GB RAM, mplib planning may OOM when combined with RoboCasa scenes

## URDF Notes

- `base_yaw_joint` is `type="revolute"` with `±2π` limits (changed from `continuous` for mplib compatibility)
- `.convex.stl` files are pre-generated for all meshes (mplib collision checking)
- Robotiq DAE meshes require `pycollada` for loading

## File Structure

```
maniskill-tidyverse/
├── tidyverse_agent.py      # Agent class, registered as 'tidyverse'
├── tidyverse.urdf           # Full URDF (base + arm + gripper)
├── franka_description/      # Symlink → mani_skill Panda meshes
├── robotiq_meshes/          # Symlink → Robotiq 2F meshes
└── README.md
```

## License

MIT
