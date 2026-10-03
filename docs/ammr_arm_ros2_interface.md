# AMMR Arm ROS2 Interface

This is the common interface used by both Isaac Sim and the future real myCobot adapter.

## Topics

| Direction | Topic | Type | Meaning |
| --- | --- | --- | --- |
| command | `/ammr/arm/joint_targets` | `std_msgs/Float64MultiArray` | `[joint1, joint2, joint3, joint4, joint5, joint6, gripper]` |
| state | `/ammr/joint_states` | `sensor_msgs/JointState` | Same 7 controllable joints with names |
| state | `/ammr/robot_state` | `std_msgs/Float64MultiArray` | Compact `[joint1..joint6, gripper]` for VLA input |
| vision | `/ammr/hand_camera/image_raw` | `sensor_msgs/Image` | Wrist RGB image |
| vision | `/ammr/hand_camera/camera_info` | `sensor_msgs/CameraInfo` | Wrist RGB camera intrinsics |

## Units

All arm joints use radians at the AMMR interface boundary.

The gripper command also uses one canonical scalar:

| Meaning | Value |
| --- | --- |
| open | `0.08` |
| object grasp close | `-0.20` |
| no-load full close debug value | `-0.35` |
| Isaac slider limit | `-0.38 .. 0.10` |

The real adapter maps the canonical gripper command range `-0.20 .. 0.08` to Elephant Robotics gripper value `0 .. 100`.

## Launch

Isaac Sim adapter:

```bash
/home/autolab/AMMR/scripts/run_mycobot_ros2_gui.sh
```

Real myCobot adapter dry-run:

```bash
/home/autolab/AMMR/scripts/run_real_mycobot_adapter.sh
```

Real myCobot adapter with hardware:

```bash
/home/autolab/AMMR/scripts/run_real_mycobot_adapter.sh --real --port /dev/ttyUSB0
```

Common command test node:

```bash
/home/autolab/AMMR/scripts/run_ammr_arm_command_demo.sh
```

Small demo sequence:

```bash
/home/autolab/AMMR/scripts/run_ammr_arm_command_demo.sh --mode demo
```

Scripted red cube pick-only demo:

```bash
/home/autolab/AMMR/scripts/run_scripted_pick_demo.sh --object red_cube --close -0.25
```

Record one raw SmolVLA episode:

```bash
/home/autolab/AMMR/scripts/run_record_smolvla_episode.sh \
  --instruction "pick the red cube" \
  --task-name pick_red_cube \
  --duration-sec 20
```

Validate a recorded episode:

```bash
/home/autolab/AMMR/scripts/run_validate_smolvla_episode.sh \
  /home/autolab/AMMR/data/smolvla_raw/episode_000001
```

Raw episode layout:

```text
/home/autolab/AMMR/data/smolvla_raw/
  episode_000001/
    metadata.json
    images/
      000000.png
    robot_state.npy
    action.npy
    timestamps.npy
```

## Policy

SmolVLA and task-level code should only use the `/ammr/...` topics above. Isaac Sim code and real robot serial code stay behind adapters.
