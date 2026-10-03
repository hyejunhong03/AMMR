# Dynamic Contact Grasp Mode

This mode is for physics debugging only. It keeps the stable grasp-assist data
collection path unchanged.

## Stable Default

Default task object behavior remains:

```text
object rigid body: kinematic
object collision: disabled
grasp assist: enabled
gripper drive: controller joint only, mimic followers passive
```

Use this path for current VLA dataset collection and education demos.

## Dynamic Contact Experiment

Enable full object dynamics explicitly:

```bash
/home/autolab/isaacsim/python.sh \
  /home/autolab/AMMR/scripts/control_mycobot_sliders_gui.py \
  --headless \
  --dynamic-contact-grasp \
  --physics-diagnostics /tmp/ammr_dynamic_contact_diag.csv \
  --physics-diagnostics-stride 10
```

This changes task object behavior to:

```text
object rigid body: dynamic
object collision: enabled
grasp assist: forced off
gripper drive: controller joint only
mimic followers: passive
```

The mode rejects incompatible gripper options:

```text
--kinematic-gripper
--simple-gripper
--drive-all-gripper-joints
```

## Reset Behavior

In dynamic-contact mode, reset and `/ammr/set_object_pose` preserve:

```text
kinematicEnabled = false
collisionEnabled = true
```

The object pose is reset through `RigidPrim.set_world_poses()` when possible, and
linear/angular velocity is zeroed through `RigidPrim.set_velocities()`.

In stable default mode, reset preserves the previous kinematic/non-colliding
behavior.

## Diagnostics CSV

`--physics-diagnostics` records inside the simulator loop, not through ROS2 topic
sampling. With `--physics-diagnostics-stride 1`, every physics step is recorded.

The CSV includes:

```text
step, sim_time_s, wall_dt_s, physics_dt_s
joint position / velocity / position target for every DOF
max arm velocity
max control target error
gripper mimic follower error
object position
object linear/angular velocity
object speed
object contact_force_norm
```

`object.*.contact_force_norm` is currently `NaN`. Contact force logging is left for
the collider/material tuning pass because the correct filter prims need to be
defined after the gripper contact geometry is finalized.

## Verified Smoke Test

The following smoke test was run successfully:

```bash
timeout 30s /home/autolab/isaacsim/python.sh \
  /home/autolab/AMMR/scripts/control_mycobot_sliders_gui.py \
  --headless \
  --no-wrist-camera \
  --dynamic-contact-grasp \
  --physics-diagnostics /tmp/ammr_dynamic_contact_diag.csv \
  --physics-diagnostics-stride 10 \
  --settle-time 0.1 \
  --loop-profile 5
```

Result:

```text
startup: OK
dynamic-contact mode: OK
diagnostics CSV: OK
CSV rows: 845 data rows plus header
contact force columns: NaN by design
```

This smoke test does not validate successful physical grasp yet. It only verifies
that the separated dynamic-contact experiment path and the internal diagnostics
logger run without breaking the stable default path.
