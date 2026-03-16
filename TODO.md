# TODO

## Bugs

- [ ] Loading a saved Blockly program flips "grabber close" to "grabber open" when the program was saved with the old dropdown value (150 vs 140). Fix is written (field migration in blockly_hook.js) but not yet deployed.

## Features

### Web dashboard
- [ ] Camera feed on the dashboard (WebSocket binary frames → `<img>` or `<canvas>`)
- [ ] Mode switcher UI (manual / sonar avoidance / infrared line-following)
- [ ] LED effect controls (color wipe, breathe, rainbow) — currently only static color + off
- [ ] Camera tilt servo control (channel 2) on dashboard
- [ ] Mobile-friendly touch controls (virtual joystick?)
- [ ] Connection status indicator (robot online/offline)

### Block programming
- [ ] More Blockly blocks: repeat-while-distance, follow-line, camera-tilt
- [ ] Export program as shareable JSON (not just localStorage)
- [ ] Import program from file
- [ ] Example programs (obstacle avoidance, square patrol, grab-and-deliver)

### Robot server
- [ ] Systemd service file for auto-start on boot
- [ ] Watchdog / auto-restart on crash
- [ ] Config file for servo limits (instead of hardcoded in protocol.py)
- [ ] Telemetry logging to file for post-session analysis
- [ ] Battery voltage monitoring (ADC if available)

### Vision / Autonomy
- [ ] Streaming annotated video (bounding boxes) back to web dashboard
- [ ] Multiple navigation strategies (explore, follow person, patrol waypoints)
- [ ] Model selection (YOLOv11 nano/small/medium depending on GPU)
- [ ] Record and replay autonomous sessions

### Infrastructure
- [ ] CI pipeline (lint + typecheck on push)
- [ ] Docker compose for web + vision (desktop side)
- [ ] OTA update mechanism for Pi (beyond rsync)
- [ ] Proper logging (structured, rotated) on Pi

## Done (recent)

- [x] WASD keyboard controls with additive multi-key support
- [x] Arm and grabber servo controls (buttons + keyboard R/F/T/G)
- [x] Hold-to-repeat for servo buttons (120ms interval)
- [x] Motor duty ramping (400/step, 50 Hz) to prevent undervoltage
- [x] Servo smooth sweeping (3°/step, 50 Hz)
- [x] Servo hold-on-shutdown (no arm drop)
- [x] Blockly visual programming with custom robot blocks
- [x] Blockly: built-in blocks (logic, loops, math, variables, text)
- [x] Blockly: "When program starts" hat block
- [x] Blockly: save/load/delete programs (localStorage)
- [x] Blockly: autosave every 2 seconds
- [x] Blockly: dark theme, zero-DOM-patch architecture
- [x] Camera flip (hflip + vflip) for upside-down mount
- [x] Tuned servo limits (grabber 90-140°, arm 75-150°)
- [x] Legacy protocol compatibility (Freenove Android app)
- [x] WebSocket API (JSON commands + binary video)
- [x] Per-client WebSocket send queues
- [x] Desktop vision engine (YOLOv11 + obstacle avoidance)
- [x] Taskfile automation (setup, deploy, dev, lint, typecheck)
