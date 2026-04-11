# Autonomy Perception Redesign Plan

**Status:** proposal, pre-implementation
**Date:** 2026-04-11
**Scope:** replace the MASt3R-SLAM-centric live autonomy stack with a layered, pose-quality-gated perception architecture

---

## 1. Problem statement

The current autonomy stack depends on monocular MASt3R-SLAM for both pose estimation and primary obstacle perception (`src/tankbot/desktop/autonomy/slam.py`, `src/tankbot/desktop/autonomy/vision.py`). This couples "don't crash" to "maintain a globally consistent map" — when visual tracking degrades, localization and depth degrade together, and safety breaks.

The failure mode is visible in the control code itself. `vision.py` carries `UNSTABLE_TRACKING_FRAME_LIMIT`, `PARALLAX_PROBE_*`, pulse-and-coast driving (`PULSE_DURATION = 0.20`, `COAST_PAUSE = 0.08`), settle delays after turns, and multi-stage recovery logic — all existing to keep MASt3R's visual tracker alive. The perception stack is dictating robot behavior instead of serving it.

Two structural reasons this is not solvable by tuning MASt3R harder:

1. **Tanks turn in place.** Pure rotation produces zero translational parallax, which is monocular SLAM's worst failure case. Every in-place turn is a guaranteed tracking stress event regardless of which monocular backend is used.
2. **Monocular scale is unobservable.** Today's stack calibrates MASt3R depth against ultrasonic, which is a fragile coupling — especially on the soft surfaces where ultrasonic itself is unreliable (see `feedback_ultrasonic_unreliable.md`).

## 2. Guiding principles

1. **Decouple "don't crash" from "build a consistent map."** Safety must not depend on a long-horizon tracker staying alive.
2. **Layer autonomy by temporal horizon and pose requirements.** Each layer degrades gracefully to the layer below it.
3. **Pose quality gates planner participation, not the other way around.** No pose source drives the persistent planner before it has earned the right on replay data.
4. **Validation requirements scale with layer.** Persistent world-frame planning requires replay validation against a curated set with independent ground truth. Per-frame reactive behavior — which has no accumulating state that can poison decisions — may ship live after a defined live-test protocol (supervised, bounded environment, measurable behavioral criteria). "Validate before live" is not a universal principle; it applies to layers whose errors compound, not to stateless per-frame vetoes.
5. **Introduce abstractions, not drop-in swaps.** Depth source, pose source, and mapping layers are separate concerns with separate lifecycles. Abstractions are defined concretely *before* they have multiple implementations, not retrofitted.
6. **Performance is a correctness property, not a tuning concern.** A "reactive" layer that runs at 3 Hz is not reactive — it's a slow predictor with a misleading name. Throughput and worst-case latency are gated like any other requirement.

## 3. Target architecture

Three layers, each independently useful:

| Layer | Horizon | Pose requirement | Failure mode |
|---|---|---|---|
| Reactive safety grid | current frame + ~300–500 ms persistence | gyro-only heading (no translation); extrinsics + floor plane | bounded, self-clearing, cannot fail silently |
| Rolling frontier grid | tens of seconds, ~12×12 m window | fused `PoseEstimate` with health gating | degrades by health state |
| Global consistency (optional) | whole mission | mono-inertial VIO, encoders, or AprilTag anchors | degrades to rolling grid |

Key property: **each layer works when the layers above it fail.** Lose global VIO? Frontier planner drops to rolling grid or disables. Lose scan-match translation? Planner drops to BROKEN, reactive layer still vetoes bad commands. Lose depth inference for a frame? Reactive layer holds previous grid with confidence decay, then falls to ultrasonic-only hard stop.

### 3.1 Reactive safety layer

- **Input:** per-frame metric depth (see §3.1.1 for depth model — Depth Pro was an assumption in earlier drafts and is not currently integrated in this repo) + ultrasonic.
- **Output:** robot-centric short-memory grid + immediate motor command vetoes.
- **Dependencies:** camera extrinsics (one-time calibration), floor plane assumption, gyro-only heading for rotation compensation during the persistence window. **No translation estimator required.**
- **Independent of global pose.** This layer does not know where the robot is in the world frame. It only needs "how much have I rotated since the last depth frame" to keep the local grid consistent across in-place turns.
- **Grid sizing and persistence:** grid window ~4×4 m robot-centric. Persistence window tied to the depth model's achieved rate (§3.1.1); at 15 Hz target rate, cells decay over ~300–500 ms. Below the minimum control rate (§3.1.1), the grid is cleared every tick — no persistence — and the reactive layer enforces a hard speed cap.
- **In-place turn handling:** gyro integrates rotation at high rate (kHz) between depth frames. Persisted grid cells are rotated in-place around the robot origin by the gyro-reported delta before being merged with new observations. Translation between frames is assumed zero within the persistence window — valid at the reactive layer's speed cap and persistence horizon, not valid in general.
- This is the floor of safety: if everything else in the stack fails, this keeps the robot from driving into walls.

#### 3.1.1 Monocular depth integration (Phase 0b prerequisite)

This is net-new work, not a drop-in. The repo has no monocular depth model today; the only depth source in live code is MASt3R-SLAM's dense output (`pyproject.toml` desktop deps confirm this).

**Candidates to evaluate** on the target desktop GPU:
- **Depth Pro** — highest quality, ~1B params, highest latency. Metric by design.
- **Depth Anything V2 (metric variant)** — smaller, faster, good quality. Metric variant requires the indoor/outdoor split.
- **UniDepth** — metric monocular, middle ground on size/speed.

**Performance gate** (Phase 0b exit criterion):
- **Target end-to-end decision loop rate:** ≥15 Hz sustained.
- **Floor:** 10 Hz. Below the floor, reactive layer enters degraded mode with hard speed cap.
- **Worst-case latency per frame:** ≤120 ms from frame capture to motor command.
- **GPU memory budget:** must coexist with MASt3R-SLAM offline reconstruction (Phase 6) on the same desktop box.
- **Deadline-miss fallback:** if depth inference misses its deadline on a given frame, reactive layer holds the previous depth grid for up to N ticks (with confidence decay), then falls back to ultrasonic-only veto and hard-stops forward motion.

**Loop architecture decision** required in Phase 0b: keep `_decision_loop` serialized (simpler; depth FPS = control FPS) or split into a fast reactive tick consuming the most recent available depth output from a slower producer thread (more robust; decouples control rate from inference rate). Decision gated on measured depth FPS — if the chosen model hits ≥20 Hz, serialized is fine; otherwise split.

### 3.2 Fused pose source (abstraction)

A concrete `PoseEstimate` contract consumable by any layer that needs pose. Defined once, before any implementation, so the abstraction does not churn as implementations land.

```python
@dataclass(frozen=True)
class PoseEstimate:
    # Capture timestamp of the sensor readings this estimate was derived from.
    # Monotonic clock, seconds. Consumers use this to align with depth frames.
    t_monotonic: float

    # Planar pose in a 2D world frame anchored at the robot's init pose.
    # Yaw is continuous (unwrapped, not wrapped to [-pi, pi]), so planner code
    # does not have to special-case wraparound. Always valid from init onward.
    yaw_rad: float
    yaw_var: float                # rad^2, 1-sigma^2 uncertainty

    # Absolute xy since init, meters. None when no translation estimator
    # is online or the current estimator has fallen back to unknown.
    # Incremental deltas are derived by differencing, not primitive.
    xy_m: tuple[float, float] | None
    xy_var: float | None          # m^2, isotropic 1-sigma^2 for v1

    # Health state drives consumer behavior (see §3.4).
    health: HealthState           # HEALTHY | DEGRADED | BROKEN

    # Provenance — which estimator produced this. Useful for logging,
    # replay analysis, and consumer heuristics (e.g., planner can prefer
    # VIO output over scan-match output when both are available).
    source: str                   # "gyro", "scan_match", "encoders", "vio", "apriltag"
```

**Contract notes:**
- **2D only.** Ground robot on a planar floor. Camera height and pitch are static-ish and come from one-time extrinsic calibration, not from the pose layer. Chassis pitch during ramp transitions is a V2 feature handled at the depth-projection layer, not here.
- **Absolute since init**, not incremental. Incremental is error-prone at consumer sites (easy to double-integrate or skip ticks); diffing two absolute estimates is not.
- **Continuous yaw**, not wrapped. Planner frontier-selection code should never have to think about wraparound.
- **`xy_m=None` is a first-class state**, not a sentinel value like `(0, 0)`. It means "no translation estimator is online"; consumers that need translation must explicitly handle it.
- **Health is per-estimate**, not a global flag, so the pose source can degrade on a per-tick basis. A single noisy frame can produce a DEGRADED estimate without tearing down the estimator.
- **Uncertainty is isotropic in v1.** Full 3×3 covariance is deferred until there's a consumer that actually uses it.

Implementations, in order of escalation:

1. **Gyro-only.** Heading from IMU gyro integration, `xy_m=None`, `health=HEALTHY` whenever the IMU is producing samples on schedule. Translation unavailability is communicated by `xy_m=None`, not by a special health state — consumers that need translation must explicitly handle the `None` case. Enough to unblock Phase 2 without lying about translation.
2. **Gyro + depth-based 2D scan matching.** Project per-frame depth into a synthetic 2D laser scan (nearest obstacle per angular bin from a floor-parallel slice), ICP against the previous frame's scan with gyro as a rotation prior. Translation comes from the matched transform. This is the leading candidate because it uses data we already have and degrades visibly.
3. **+ wheel encoders** once hardware lands. Fused as a second independent translation channel.
4. **Mono-inertial VIO** (ORB-SLAM3 inertial or similar) as a premium pose source. The IMU participates as a constraint on the visual factor graph, **not** as a standalone translation integrator.
5. **AprilTag absolute anchors** as discrete pose corrections when a tag is visible in the operating area.

**Non-candidates:**

- ~~Commanded motor velocity as translation estimator.~~ Skid-steer tread slip, `MIN_DUTY=1000` deadband, and battery sag make this unusable. Even with encoders, skid-steer translation is noisy; commanded-only is worse.
- ~~IMU-preintegrated velocity as standalone translation estimator.~~ Double-integrated MEMS accelerometer on a vibrating platform produces noise-squared drift that grows to meters in seconds without ZUPT, encoders, or a tightly-coupled visual backend. IMU is a **heading sensor**, full stop.
- ~~Monocular-only ORB-SLAM3 as a swap for MASt3R.~~ Same pure-rotation failure mode. If ORB-SLAM3 is used, it must be mono-inertial.

### 3.3 Rolling frontier grid

Replacement for the current fixed 24×24 m world-frame grid in `frontier.py`. Key differences from the current implementation:

- **Rolling**, not accumulating-forever. Window translates with the robot; cells outside the window are dropped. ~12×12 m at 10 cm resolution is the starting proposal.
- **Bounded drift.** Old cells age out naturally, so a single bad pose frame cannot permanently corrupt the map.
- **Explicit pose-health gating,** see §3.4.
- **Frontier clustering, target selection, and `PlannerCommand` output contract stay the same.** The refactor is upstream of `FrontierPlanner`'s public API where possible.

Refactor scope in `frontier.py`:
- Replace fixed `GRID_SIZE=240` + fixed-world cell addressing with a rolling window anchored to the robot's current fused-pose position.
- Rewrite `_world_to_cell` / `_cell_to_world` to operate in window-relative coordinates with a known window-origin offset.
- `update_from_frame` consumes depth-projected world-frame points from the depth pipeline + fused pose, not SLAM's `planning_points`.
- `pose_valid` / `tracking_lost` become outputs of the pose-health model (§3.4), not SLAM-derived flags.

### 3.4 Pose-health model

Three discrete states with explicit transition criteria. The criteria below are starting thresholds; exact values come from Phase 0c replay tuning.

**HEALTHY** — read/write rolling grid, planner fully on, all frontier targets eligible.
- Entry requires **all** of:
  - scan-match residual below `τ_resid` for at least `N_healthy` consecutive frames
  - gyro bias estimate below `τ_bias`
  - translation confidence above `τ_trans`
  - time-since-last-anchor below `T_anchor_healthy` *or* encoder/VIO pose channel available

**DEGRADED** — no grid writes; planner may only **complete an in-flight very-short-horizon command** (sub-second, ~20 cm of motion) *if and only if*:
- heading (gyro) is still good
- the current target remains inside the local window
- the command's geometry is still sensible (target still roughly in the commanded direction)

After the in-flight command completes, fall through to reactive. No new frontier selection, no re-planning. DEGRADED is a **graceful handoff state, not a lingering mode.** It exists so a single noisy frame doesn't cause a jerky bail-out; it does not exist to let the planner keep driving on stale pose.

**BROKEN** — planner fully disabled, reactive layer only. Entered on sustained health failure, translation-source fallback, or any pose-sanity check tripping (e.g., implied velocity exceeds physical plausibility).

Recovery: BROKEN → DEGRADED → HEALTHY requires re-meeting the entry criteria for the higher state over a stable window. No automatic demotion-to-promotion oscillation.

## 4. Phased delivery

### Phase 0a — Dataset format + recorder/replay upgrade

**Why first:** without a synchronized multi-channel dataset, "validate on replay" can only compare vision-only candidates. We cannot evaluate anything that consumes IMU, ultrasonic, motor commands, or encoders.

**Deliverables:**
- Extend `record_robot_stream.py` to capture a synchronized, timestamped stream. Currently it discards telemetry (`self._client.on_telemetry(lambda _: None)` at line 32) and saves only PNG frames.
- New dataset format: directory of `NNNNNN.png` + `telemetry.jsonl` with one record per tick, plus a versioned `manifest.json` describing the schema. Appendable, streamable, no full-load-into-RAM requirement.
- Channels to record:
  - camera frames + monotonic timestamps
  - ultrasonic readings
  - motor commands issued (for dead-reckoning comparison baselines, not as a translation estimator)
  - IMU samples (once hardware is wired)
  - encoder ticks (once hardware is wired)
- Extend `replay_slam_dataset.py` (or add a sibling `replay_pipeline.py`) to iterate the synchronized stream and expose each channel to arbitrary candidate pipelines, not just `SplatSLAM`.

**Behavior change:** none. Pure tooling.

**Exit criterion:** a recorded dataset round-trips cleanly through the new replay harness and all channels are accessible at correct timestamps.

### Phase 0b — Monocular depth integration + performance gate

**Why before Phase 1:** the previous draft assumed Depth Pro was already integrated. It isn't. No monocular depth model exists in this repo, and the `_decision_loop` is fully serialized — depth inference latency directly becomes control-loop latency. Phase 1 cannot ship a live reactive layer until we know (a) which depth model we're using, and (b) whether it's fast enough to serve reactive control.

**Deliverables:**
- Integrate a monocular metric depth model as a desktop dependency. Candidates: **Depth Pro**, **Depth Anything V2 (metric)**, **UniDepth**. Pick one after benchmarking all three on the target GPU with representative `recordings/mast3r_raw/` frames.
- Benchmark on target hardware. For each candidate, measure:
  - Sustained FPS at 640×480 input, single-frame inference (no batching).
  - p50 / p95 / p99 per-frame latency.
  - Peak GPU memory footprint.
  - Quality sanity-check: does the depth map look right on a handful of known-geometry test scenes (a person at known distance, a wall at known distance)?
- **Gate criteria (must pass to exit Phase 0b):**
  - Sustained end-to-end decision-loop rate **≥10 Hz** (floor) with **≥15 Hz** target.
  - Worst-case per-frame latency **≤120 ms** (p99).
  - GPU memory headroom such that the depth model + MASt3R offline reconstruction can both exist on the same box (not necessarily simultaneously — MASt3R is Phase 6 offline tooling).
- **Loop architecture decision:** if the chosen model hits ≥20 Hz on its own, keep `_decision_loop` serialized. Otherwise split into a fast reactive tick consuming the most recent available depth output from a slower producer thread.
- **Deadline-miss policy:** document the exact fallback behavior when inference overruns its deadline (reactive holds previous depth with confidence decay for N ticks, then falls back to ultrasonic-only + hard forward stop).

**Behavior change:** none; this phase does not wire into live control. Its output is a chosen model, an inference wrapper, and documented measurements against the gate.

**Exit criterion:** one monocular depth model selected, benchmarked, wrapped, and demonstrated to meet the gate on the target hardware. Phase 1 cannot start otherwise — or must scope itself to ultrasonic-only, which is probably not worth shipping.

#### Phase 0b measured results (2026-04-11)

Hardware: RTX 4070 Ti SUPER (16 GiB), torch 2.10+cu128, 640×480 serial inference, 15 warmup / 150 measured iterations, CUDA-synced per frame, `recordings/mast3r_raw/vanilla1`. Full JSON reports in `logs/depth_bench/phase0b_base/`.

| Backend | Checkpoint | Mean ms | p99 ms | Sustained FPS | Peak GPU MiB | Verdict |
|---|---|---|---|---|---|---|
| Depth Anything V2 | `…Metric-Indoor-Small-hf` | 17.9 | 19.3 | 55.9 | 245 | **PASS** |
| Depth Anything V2 | `…Metric-Indoor-Base-hf`  | 43.3 | 45.7 | 23.1 | 615 | **PASS** |
| Depth Anything V2 | `…Metric-Indoor-Large-hf` | 125.9 | 130.2 | 7.9 | 1711 | FAIL (latency + FPS) |
| Depth Pro | `apple/DepthPro-hf` | 340.1 | 345.1 | 2.9 | 3776 | FAIL (large margin) |
| UniDepth | `lpiccinelli/unidepth-v2-vitl14` | — | — | — | — | deferred (upstream pkg not installed) |

**Selection:** Depth Anything V2 **Base** (`depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf`). Base is the default for Phase 1. It clears the 15 Hz target with ~8 ms headroom per frame at 23 Hz sustained and keeps peak GPU under 620 MiB, leaving ample room for MASt3R offline reconstruction to co-exist. Median reported depth on an indoor hallway frame was 1.85 m (min 0.81, max 6.48), consistent with the scene geometry — a decent metric-sanity check pending Phase 0c tape-measured ground truth.

**Loop architecture decision:** Base inference is ~43 ms on this GPU, which is above the 20 Hz threshold the plan set for keeping `_decision_loop` serialized (≥ 20 Hz → serialized is fine). We land Phase 1 with the serialized loop and revisit a fast-reactive/slow-producer split only if camera + preprocessing headroom shrinks below the 10 Hz floor in practice. Small variant (56 Hz) is the fallback if Base proves too slow in the live pipeline end-to-end.

**Depth Pro verdict:** single-shot inference is ~12× too slow for the reactive tick on this hardware, and peak GPU memory is ~6× higher than Base. Quality is not disputed, but performance rules it out for live control. It remains a viable offline labeler for Phase 0c ground-truth augmentation if needed.

**UniDepth verdict:** deferred. Upstream package is not on PyPI and its CUDA-op install path is fragile. Not a blocker: Depth Anything V2 Base already passes the gate, and there is no quality-driven reason to chase UniDepth unless Base fails on a downstream quality metric in Phase 0c.

**Benchmark harness:** `tankbot-benchmark-depth --dataset <path> --backends depth_anything_v2 depth_pro unidepth`. Per-backend checkpoint is overridable via `TANKBOT_DEPTH_ANYTHING_V2_CHECKPOINT` / `TANKBOT_DEPTH_PRO_CHECKPOINT` / `TANKBOT_UNIDEPTH_CHECKPOINT` env vars so re-benchmarking different variants does not require code changes. Reports land in `logs/depth_bench/<timestamp>/` as one JSON per backend plus a `summary.json`.

### Phase 0c — Curate replay set + metrics

**Prerequisite:** Phase 0a landed (the harness can read multi-channel data) and at least a few representative multi-channel recordings exist.

**Deliverables:**
- Small curated set of "known hard" scenarios: in-place turns, textureless wall approach, dim corridor, low obstacle near the tracks, person walking past, door-threshold chassis pitch, rug-to-hardwood transition.
- **Ground truth with independent references** (not the candidate sensors themselves):
  - **Tape-measured fixture recordings.** Place a known obstacle at marked distances (0.25, 0.5, 1.0, 2.0 m) from a marked robot position, record a short clip at each. Use these for absolute distance-error measurements on any depth candidate.
  - **Hand-annotated "stop-zone occupied" labels.** For each frame in the curated set, a human labels whether there is an obstacle inside the robot's stop zone (a fixed cone/wedge in front). Boolean, cheap to annotate, directly maps to the property we actually care about.
  - **Tape-measured trajectory endpoints.** For translation-candidate evaluation, record trajectories where the robot starts and ends at marked floor positions. Measure each candidate's reported start→end displacement against the tape measurement.
  - **Ultrasonic cross-check on hard flat surfaces only** — useful as a sanity signal, never used as ground truth where ultrasonic is unreliable (soft surfaces, see `feedback_ultrasonic_unreliable.md`).
- Metric definitions:
  - **Primary (safety):** "stop-zone occupied" precision and recall against the hand labels. This is the metric that matters for the reactive layer.
  - **Secondary (accuracy):** absolute distance error at the tape-measured fixtures. Reported as median + p95 per candidate.
  - **Pose-source metric:** endpoint displacement error in meters and final-heading error in degrees on the tape-measured trajectories. Reported per candidate over the curated set.
  - **Qualitative replay:** when (eventually) the rolling frontier grid exists, does it chase phantom frontiers on the known-hard segments?
- Rule: **no candidate ships to live control until it passes the primary metric on the curated set.**

**Note on circularity:** no metric in this phase uses any candidate's own output as its ground truth. The previous draft's "per-frame obstacle distance error vs. Depth Pro ground truth" was circular and has been removed.

**Exit criterion:** replay harness + ground-truth recordings + metric evaluation are reproducible from one command; the current MASt3R baseline is measured for before/after comparison.

### Phase 1 — Reactive safety layer, FrontierPlanner disabled

**Prerequisites:** Phase 0b passed (depth model selected and meets the performance gate); IMU hardware present and gyro driver available. Phase 0a and 0c are **not** prerequisites for Phase 1 — Phase 1 is a live-shippable reactive improvement, not a validation phase. Phase 0c's replay set is used as a regression check before and after Phase 1 but does not gate it.

**Scope clarification:** Phase 1 is a **shippable safety improvement**, not a live experiment. It replaces the current MASt3R-dependent control stack with something strictly simpler and more bounded. It ships under a defined live-test protocol, not under the "validate on replay first" rule, because its errors do not compound — every frame is a fresh decision.

**Deliverables:**
- Wire the Phase 0b-selected depth model into `_decision_loop` with the chosen loop architecture (serialized or split).
- **Define the `PoseEstimate` dataclass from §3.2** (this is the moment the abstraction lands — not deferred to a later phase).
- **First `PoseSource` implementation: gyro-only.** IMU driver, gyro integration, `yaw_rad` + `yaw_var`, `xy_m=None`, `source="gyro"`. Health is always HEALTHY while the IMU is producing samples. This is the minimum ego-motion Phase 1 needs for rotation compensation in the persistence window.
- **Robot-centric short-memory safety grid** (~4×4 m, ~300–500 ms persistence at 15 Hz target). Cells are rotated in place by gyro deltas between depth frames. Translation assumed zero within the persistence window — enforced by the speed cap below.
- **Floor-plane projection** using one-time camera extrinsic calibration.
- **Ultrasonic fused as a short-range free-space veto / obstacle confirmation,** not as a scale anchor.
- **Explicitly remove `FrontierPlanner` from the live control loop wiring.** Not gated by a flag — *deleted from the loop*. A disabled-by-flag planner invites re-enabling under a bad pose source.
- **Reactive exploration strategy:** simple wander (forward until blocked, turn, repeat, small random bias). Not optimal coverage, but reliable.
- **Delete SLAM-accommodation code** once MASt3R is off the live path: `PARALLAX_PROBE_*`, `UNSTABLE_TRACKING_FRAME_LIMIT`, pulse-and-coast (`PULSE_DURATION` / `COAST_PAUSE`), settle delays, recovery retreats.

**Behavioral limits (live-test protocol):**
- **Speed cap** during Phase 1: forward cruise capped such that the robot moves at most ~5 cm per persistence window. At 300 ms persistence, that's ~17 cm/s. Below the current `EXPLORE_SPEED = 1800` on purpose.
- **Degraded mode** triggers when the decision loop falls below 10 Hz sustained: speed cap drops further, grid persistence goes to zero (every tick is fresh), reactive layer enforces a hard forward stop if depth has not updated for >300 ms.
- **Supervised test environment first:** bounded area (single room, no stairs, no fragile obstacles), human e-stop within reach, at least one full session of observed behavior before unattended runs.
- **Behavioral success criteria:** (a) no collisions over N minutes of continuous operation in the bounded environment, (b) successful wander without getting stuck in a single corner longer than M seconds. A Phase 0c regression check (no regression against MASt3R baseline on the primary replay metric) is **informational only** if Phase 0c has landed by this point; it is not a Phase 1 gate. If 0c has not landed, skip the check entirely. This keeps Phase 0c a true non-blocker for Phase 1 shipment.

**Exit criterion:** robot runs continuous reactive exploration in a typical indoor environment without collisions, without the SLAM-accommodation driving workarounds, and without the `FrontierPlanner` in the loop. `vision.py` is meaningfully smaller than it is today.

### Phase 2 — Translation candidates, PoseSource upgrades

**Prerequisite:** Phase 0a, 0b, and 0c all landed; Phase 1 shipping and stable.

**Deliverables:**
- Implement depth-based 2D scan matching as a second `PoseSource` implementation: projects depth into a synthetic 2D laser scan, ICPs against the previous frame with gyro-derived rotation prior, returns `yaw_rad` from gyro and `xy_m` from the matched translation.
- Run on the Phase 0c replay set. Measure endpoint displacement error and final-heading error against the tape-measured trajectories.
- Define fallback: if scan-match confidence drops below threshold for N consecutive ticks, `xy_m` reverts to `None` and health transitions DEGRADED → BROKEN.
- If encoders have landed: implement encoder-based translation provider and a fused scan-match + encoder provider. Evaluate all candidates on the replay set. Pick the winner by measured drift, not intuition.
- Reactive layer remains `yaw`-only; translation candidates feed only into the planner refactor that comes in Phase 4.

**Exit criterion:** at least one translation candidate produces bounded drift on the curated replay set and is selected as the Phase 4 input. The chosen candidate's drift-per-meter and drift-per-turn are documented.

### Phase 3 — Rolling frontier grid refactor, planner still offline

**Prerequisite:** Phase 2 landed with a selected translation candidate.

**Deliverables:**
- Refactor `FrontierPlanner` to the rolling-window design described in §3.3. Planner stays disabled in the live loop during this phase — refactor is exercised on the replay harness only.
- Wire `PoseEstimate` (§3.2) as the input contract, replacing MASt3R camera pose + planning points.
- Implement the pose-health model (§3.4) with initial thresholds calibrated from Phase 0c data.

**Exit criterion:** refactored planner runs cleanly on Phase 0c replay set, health model transitions as expected on known-hard segments, no phantom-frontier behavior observed offline.

### Phase 4 — Rolling frontier grid re-enabled in live control

**Prerequisite:** Phase 3 replay validation passed.

**Deliverables:**
- Re-enable the refactored `FrontierPlanner` in the live decision loop, consuming the Phase 2 `PoseSource`.
- Replay validation: run the refactored planner against the curated set. Required behavior:
  - planner stays HEALTHY across representative good trajectories
  - planner drops to DEGRADED/BROKEN on trajectories where MASt3R currently loses tracking — **and does not chase phantom frontiers in those segments**
  - recovery to HEALTHY occurs within expected windows
- Only then: re-enable in live control.

**Exit criterion:** replay-validated pose-health gating + rolling grid produce useful frontier exploration in live use without the failure modes that killed the MASt3R-based planner.

### Phase 5 — Global consistency layer (optional, long horizon)

**Deliverables (any one earns the right to a wider map):**
- Mono-inertial VIO (ORB-SLAM3 with IMU) integrated as a premium `PoseSource`. The rolling grid can optionally widen when this is HEALTHY.
- OR: AprilTag anchor system. A small number of tags placed in the operating area, detected per frame, used as discrete absolute-pose corrections. Requires an explicit correction strategy: rigid-transform the rolling grid on snap, or hot-reset when delta exceeds threshold. **No snap-and-seam on persistent maps.**
- OR: wheel encoders + IMU + scan-matching fusion tuned to the point where drift is demonstrably bounded over multi-minute trajectories.

**Decision on fixed whole-environment world grid:** only revisit this question after Phase 5 lands. The current fixed-grid `FrontierPlanner` is being replaced, not restored.

### Phase 6 — MASt3R-SLAM retired from live control

**Deliverables:**
- MASt3R-SLAM removed from the live autonomy loop.
- Retained as an **offline** reconstruction tool: run it on recorded datasets to produce PLY exports for the dashboard 3D view, decoupled from control. This is what MASt3R is actually good at.
- `src/tankbot/desktop/autonomy/slam.py` either deleted or reduced to the offline-only code path.
- `vision.py` is simplified to the new layered architecture; all `PARALLAX_PROBE_*`, `UNSTABLE_TRACKING_FRAME_LIMIT`, pulse-and-coast, and MASt3R-specific recovery logic removed.

## 5. Hardware additions

Independent of the software phases:

- **IMU (MPU6050 or similar), I2C, ~$3–5.** Highest-leverage addition. Unblocks Phase 2 onward. Short driver, trivial wiring on the Pi.
- **Wheel / tread encoders, ~weekend project.** Hall-effect on motor shaft or photo-interrupter + slotted disc. Cheap, noisy on skid-steer but meaningfully better than zero. Not a software blocker but should land before Phase 3's final candidate selection.
- **AprilTags printed on paper, free.** Deployed when Phase 5 starts.
- **Optional: 2D lidar (LD06 / LD19 / YDLIDAR X2, ~$90).** Not required by this plan, but consciously noted as an alternative direction. If near-term reliability matters more than "make it work with what I have," a 2D lidar + RTAB-Map or Cartographer is the gold-standard ground-robot stack and sidesteps most of the monocular-depth risk.

## 6. Non-goals and explicit cuts

Things this plan deliberately does **not** pursue:

- **Tuning MASt3R-SLAM for live control.** The failure is structural, not parameter-driven.
- **Monocular-only ORB-SLAM3 as a drop-in replacement.** Same failure mode on a tank.
- **DROID-SLAM or DPV-SLAM as alternatives.** Even heavier than MASt3R; do not solve the pure-rotation problem.
- **End-to-end learned navigation (NoMaD, ViNT, GNM family).** Interesting but research-grade and orthogonal to the mapping question.
- **IMU double-integration for translation.** See §3.2 non-candidates.
- **Commanded-velocity dead reckoning as a translation estimator.** See §3.2 non-candidates.
- **Preserving `FrontierPlanner`'s fixed-world-grid invariant.** The refactor changes the grid model, not just its inputs.

## 7. Risks and open questions

- **Monocular depth model is net-new integration, not a drop-in.** Previous drafts incorrectly assumed Depth Pro was already in the stack; `pyproject.toml` desktop deps confirm no depth model is integrated today. Phase 0b is where this gets resolved. If no candidate meets the performance gate on the target hardware, the entire plan needs rethinking — probably toward adding a 2D lidar.
- **Depth-based scan matching is promising but unvalidated on this hardware.** Phase 0c replay is the gate; if it fails the bar, Phase 2 falls back to encoders + gyro only, and Phase 4 may need to defer until Phase 5 lands.
- **Floor-plane assumption can break on ramps, thresholds, and rug edges.** The reactive layer must handle chassis pitch gracefully or degrade to ultrasonic for those frames.
- **Monocular depth frame-to-frame consistency on textureless walls, glass, and backlit scenes is unknown on this exact camera.** Part of Phase 0b quality sanity-check plus Phase 0c replay.
- **Control loop is currently serialized.** `_decision_loop` runs analysis one frame at a time; depth inference latency becomes control latency. Phase 0b must decide whether to keep serialized or split into fast reactive + slow producer. Wrong call degrades "reactive" to "slow predictor."
- **Encoder timing on the Freenove chassis is not yet scoped.** Hall sensor placement on the existing motors may require mechanical work beyond "plug in and go."
- **Pose-health thresholds (`τ_resid`, `τ_bias`, `τ_trans`, `T_anchor_healthy`, `N_healthy`) are unknown.** Will be set from Phase 0c data, not from first principles.
- **AprilTag correction strategy is deferred to Phase 5.** Naive snap-and-accept-the-seam is unsafe for a persistent planner grid; the rolling grid tolerates it better but still needs thought.

## 8. Success criteria

The plan succeeds when:

1. The robot runs continuous indoor autonomy without MASt3R-SLAM in the live loop.
2. Tracking loss is no longer a concept the control code has to work around — there is nothing to lose.
3. `vision.py` no longer contains `PARALLAX_PROBE_*`, `UNSTABLE_TRACKING_FRAME_LIMIT`, pulse-and-coast timing, or multi-stage SLAM recovery.
4. The perception stack degrades gracefully: reactive always works, local grid works when reactive + gyro works, frontier planning works when pose health is HEALTHY, global consistency works when Phase 5 is deployed.
5. MASt3R-SLAM still produces beautiful 3D reconstructions — offline, on recorded data, decoupled from control.
