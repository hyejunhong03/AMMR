# Unseen Single-Object Catalog v0

## Purpose

`data/unseen_single_object_catalog_v0.json` is the object identity split for the
next research stage:

> single-object grasping with objects that were not included in fine-tuning.

This file is a split contract first, not yet a full runtime scene definition. The
current simulator still publishes active objects in this order:

```text
red_cube, blue_cylinder
```

Keeping the new catalog separate avoids silently changing `object_state.npy` shape
for existing red/blue datasets.

## Splits

Train objects:

- `train_green_cube_30mm`
- `train_yellow_box_flat`
- `train_purple_box_tall`
- `train_cyan_cylinder_30x50`
- `train_gray_cylinder_25x35`
- `train_teal_box_wide`

Test objects:

- `test_white_box_tall`
- `test_lime_cylinder_wide`
- `test_magenta_box_flat`
- `test_navy_cylinder_tall`

Regression objects:

- `red_cube`
- `blue_cylinder`

Regression objects are for pipeline checks only. They should not be used as the
main evidence for unseen-object generalization.

## Metadata Rule

Every generated pose cache and collected episode should preserve:

- `catalog_version`
- `object_id`
- `seen_split`
- `shape_class`
- `object_size_m` or `object_radius_m` / `object_height_m`
- `object_color_rgb`
- `object_position`

These fields are metadata only. They must not become policy inputs.

## Current Implementation State

Completed in this pass:

- Active red/blue simulator object definitions are centralized in
  `scripts/ammr_mycobot_interface.py`.
- Isaac scene creation, reset, object-state publishing, and grasp assist iterate over
  the active object list rather than duplicating red/blue code paths.
- `plan_grasp_poses.py` writes object metadata into pose caches.
- `collect_episodes.py` copies pose-cache object metadata into episode metadata.

Remaining before collecting true unseen-object episodes:

- Add a runtime scene mode that instantiates one catalog object at a time.
- Decide whether that mode publishes a generic `target_object_xyz` schema or expands
  the active object-state list.
- Add shape-specific visual creation for catalog objects beyond active red/blue.
- Validate scripted IK pick for each train/test object geometry before collecting
  policy data.
