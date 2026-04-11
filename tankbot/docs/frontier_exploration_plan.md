# Frontier-Driven Exploration Plan

## Summary

Tankbot already has a strong perception stack:
- dense MASt3R-SLAM depth
- camera pose
- keyframe / relocalization state
- accumulated 3D point cloud export

The weak part is action selection. The robot still moves mostly from short-horizon
reactive heuristics in `vision.py`:
- strip depth bias
- ultrasonic clearance
- keyframe cadence
- recovery-specific turn / undo logic

That produces motion which is locally reasonable but globally erratic. The next
architecture step is to make movement map-aware.

The first target is **coverage-oriented room exploration**, not speed.

## Main Design

### 1. Planning map layer

Add a new autonomy planning module that maintains a coarse 2.5D planning map
from SLAM output.

Inputs:
- `depth_map`
- `camera_pose`
- `pose_valid`
- `tracking_lost`
- `new_keyframe`

Derived map state:
- `free`
- `occupied`
- `unknown`
- optional `unsafe` / low-confidence cells

Implementation notes:
- Use a coarse cell size, e.g. `5-10 cm`
- Keep both:
  - a persistent world map for coverage
  - a local robot-centric crop for fast action selection
- Fuse observations conservatively using hit / free counts rather than hard
  overwrite
- Ignore updates while pose is invalid or relocalizing

### 2. Frontier extraction

Detect frontiers as unknown cells adjacent to reachable free cells.

The planner should:
- cluster frontier cells
- estimate a target viewpoint or target heading for each cluster
- score them by:
  - expected information gain
  - reachability / traversability
  - heading-change cost
  - recent tracking stability
  - anchor visibility / relocalization friendliness

The chosen frontier becomes the active exploration target until one of these
conditions holds:
- target reached
- target no longer a frontier
- progress stalls
- obstacle risk becomes too high
- tracking quality degrades

### 3. Planner command layer

The planner should not directly issue wheel PWM. It should emit a compact command
for the low-level FSM to execute.

Suggested output shape:
- `mode`: `hold | turn | forward | approach_frontier | revisit_anchor | recovery_scan`
- `target_heading`
- `target_cell` or `target_pose`
- `reason`
- `confidence`

This keeps a clean separation:
- planner decides *where* to go next
- controller decides *how* to realize the motion safely

### 4. Low-level controller responsibilities

`vision.py` should remain the low-level executor and safety layer.

It should still own:
- startup scan
- motor primitives
- obstacle stop / turn override
- stuck detection
- recovery FSM

But normal navigation should stop using direct strip-depth steering as the main
policy. Instead it should:
- ask the planner for the next command
- execute short pulses toward that target
- re-evaluate after every pulse

Reactive strip steering should remain only as:
- collision avoidance override
- local steering assist while approaching a chosen target

### 5. Recovery behavior

Recovery should be separated from exploration.

Recovery behavior should:
- stop map updates while pose is invalid
- prefer recent known-good anchor views
- choose motions that maximize re-observation of mapped structure
- only return to exploration after a short stable-tracking probation

Exploration should not continue while the system is only partially recovered.

### 6. Planner telemetry

Add planner telemetry to the existing vision status path and dashboard.

Useful fields:
- `planner_mode`
- `frontier_count`
- `selected_frontier_id`
- `coverage_ratio`
- `free_cell_count`
- `unknown_cell_count`
- `planner_reason`

This will make debugging much easier than inferring planner state from motor
commands alone.

## Phased Implementation

### Phase 1: Planning substrate

- Add a planning map module
- Project valid SLAM depth into the coarse grid
- Track free / occupied / unknown cells
- Export planner telemetry without changing navigation yet

Success criteria:
- dashboard/logs show stable frontier count and coverage growth
- planning map updates only while pose is valid

### Phase 2: Frontier selection

- Implement frontier detection and clustering
- Score candidate frontiers
- Choose one active frontier target
- Add planner command output

Success criteria:
- logs show a single selected target instead of random alternating motion
- robot prefers new room edges over already explored space

### Phase 3: Frontier-guided motion

- Replace reactive cruise steering with planner-driven target following
- Keep obstacle and recovery overrides in place
- Use short pulses and frequent replanning

Success criteria:
- robot explores the room more systematically
- reduced left/right dithering
- fewer “good map but random movement” failures

### Phase 4: Revisit / anchor behavior

- Add explicit `revisit_anchor` behavior for unstable tracking
- use recent local keyframes as candidate anchor viewpoints

Success criteria:
- recovery motions become deliberate and map-oriented
- fewer immediate re-loss events after relocalization

## Future Goal Layer

This design is intended to support a higher-level DSL later.

Planned goals:
- `ExploreRoom`
- `FindBoundary`
- `GoToPose`
- `FindObject`

Those goals should select behavior policies on top of the same planning map,
rather than each one inventing its own motor heuristics.

## Non-Goals For V1

- full global path planning
- semantic object search
- replacing MASt3R-SLAM
- perfect metrically accurate occupancy mapping

The first goal is simply:
- use the scene information we already have
- make movement purposeful
- improve room coverage
- keep recovery separate from exploration
