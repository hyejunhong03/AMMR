# Physics Tuning Requested Materials Response

## Generated Logs

The following logs were generated on 2026-09-22 from the current AMMR Isaac Sim setup:

```text
/home/autolab/AMMR/physics_tuning_isaac_run.log
/home/autolab/AMMR/physics_tuning_gripper_only.log
/home/autolab/AMMR/physics_tuning_scripted_pick.log
```

Notes:

- `physics_tuning_isaac_run.log` includes Isaac Sim startup, robot/USD loading, drive settings, ROS2 bridge setup, grasp-assist settings, loop profiling, and the red-cube attach event.
- `physics_tuning_gripper_only.log` runs gripper open/close without object contact.
- `physics_tuning_scripted_pick.log` runs a red-cube scripted pick.
- The final shutdown lines in `physics_tuning_isaac_run.log` are from manually terminating the logging run after the test. They are not part of the pick failure or physics instability.

## Isaac Sim Version

```text
6.1.0-rc.26+release.49347.2d230af4.gl
```

## Exact Gripper Model

Current simulator model:

```text
myCobot 280 M5 + Elephant Robotics adaptive gripper
```


The current tuned setting drives only `gripper_controller`.
The five mimic follower joints are passive at runtime.

Physical product link/photo and whether the real fingertips have rubber pads should be confirmed by the hardware owner.

## Object Conditions

Current simulation object used for the log/video:

```text
object: red_cube
shape: cube
size: 0.03 m x 0.03 m x 0.03 m
mass: 0.03 kg
initial center position: (0.24, -0.063, 0.115) m
table top height: 0.10 m
object center z: 0.115 m
```

Important physics note:

```text
The pick target is kinematic and its collision is disabled in the current stable setup.
The object is attached to the gripper by grasp-assist logic after the gripper closes near the target.
This was done to avoid joint spikes caused by unstable gripper/object contact dynamics.
```

So the current result should be interpreted as a simulator-stable grasp-assist setup, not a full dynamic-contact grasp.

## Target Conditions

Current target condition:

```text
Simulation-only validation first.
Robot arm base is fixed during pick.
Mobile base is not moving during the pick action.
The same AMMR interface is intended to be reused later for the real myCobot.
```

Planned real-robot transfer:

```text
Yes, eventual transfer to the real myCobot is planned, but current logs are Isaac Sim validation logs.
For real hardware, a separate velocity/target-vs-state safety limiter should be added before running learned policies.
```

## Wrist Camera and Bracket Weight

Current simulator:

```text
wrist RGB camera is modeled as a camera sensor attached near the wrist.
It is not currently modeled as a separate physical mass/inertia payload.
camera resolution: 256 x 256
ROS2 image topic: /ammr/hand_camera/image_raw
```

The real wrist camera plus bracket combined weight is not yet recorded and should be measured separately before real-robot tuning.

## Collision Visualization Screen

The current stable setup intentionally disables collision on the red cube, so a collision-debug view of the target object may be misleading.

Useful visual evidence instead:

```text
1. stationary robot
2. gripper open/close without object contact
3. scripted red-cube pick
4. terminal logs showing grasp-assist attach
5. gripper controller-only drive setting and passive mimic followers
```

If collision visualization is still required, it should focus on the robot/gripper collision geometry, not on the red cube grasp target.
