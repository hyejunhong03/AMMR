# AMMR Physics Tuning Handoff Info

## 1. Isaac Sim Version

Current local Isaac Sim version:

```text
6.1.0-rc.26+release.49347.2d230af4.gl
```

Version source:

```bash
cat /home/autolab/isaacsim/VERSION
```

## 2. Current Execution Command and Runtime Settings

Main launch script:

```bash
/home/autolab/AMMR/scripts/run_mycobot_ros2_gui.sh
```

That script runs:

```bash
/home/autolab/isaacsim/python.sh \
  /home/autolab/AMMR/scripts/control_mycobot_sliders_gui.py \
  --ros2 \
  --view-wrist-camera
```

Current default robot USD:

```text
/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/mycobot_280_m5_adaptive_gripper/mycobot_280_m5_adaptive_gripper.usda
```

Current default runtime physics settings in `scripts/control_mycobot_sliders_gui.py`:

```text
physics_hz                 240.0
render_hz                  60.0
arm_drive_stiffness         120.0
arm_drive_damping            24.0
arm_max_effort               60.0
arm_max_velocity              0.5 rad/s
gripper_stiffness           200.0
gripper_damping              20.0
gripper_max_effort            5.0
gripper_max_velocity          3.0 rad/s
gripper_open                  0.08 rad
gripper_close                -0.18 rad
grasp_attach_threshold       -0.16 rad
grasp_engage_threshold       -0.16 rad
grasp_release_threshold       0.02 rad
grasp_attach_distance         0.03 m
grasp_still_time              0.2 s
grasp_still_tolerance         0.003 rad
```

Current gripper command mode:

```text
Default: drive only gripper_controller.
Mimic follower gripper joints are made passive at runtime.
Legacy --drive-all-gripper-joints is available only for comparison.
```

Recommended command for physics tuning log capture:

```bash
cd /home/autolab/AMMR
./scripts/run_mycobot_ros2_gui.sh \
  --ros2-state-hz 30 \
  --ros2-image-hz 15 \
  --hold-gripper-after-attach \
  --loop-profile 10 \
  2>&1 | tee /home/autolab/AMMR/physics_tuning_isaac_run.log
```

Recommended scripted pick command in a second terminal:

```bash
cd /home/autolab/AMMR
python3 scripts/scripted_pick_demo.py \
  --demo-pose p1 \
  --object red_cube \
  --poses-file data/grasp_poses_red_cube_focus.json \
  --no-check-cube \
  2>&1 | tee /home/autolab/AMMR/physics_tuning_scripted_pick.log
```

Recommended gripper-only spike test:

```bash
cd /home/autolab/AMMR
python3 scripts/test_gripper_only_spike_ros2.py \
  2>&1 | tee /home/autolab/AMMR/physics_tuning_gripper_only.log
```

## 3. Current Motion Video Request

