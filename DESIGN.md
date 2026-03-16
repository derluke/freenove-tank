# Design Principles

Guiding decisions for the tankbot codebase.

## 1. Keep the Pi lightweight

The Raspberry Pi runs only hardware drivers and protocol servers — no UI, no ML inference, no web server. Heavy work (vision, web dashboard) runs on the desktop and talks to the Pi over WebSocket.

**Why:** The Pi 5 has limited RAM, limited GPU, and is powered by batteries. Moving compute off the Pi keeps it stable and responsive. It also means the robot works without a desktop — the Freenove Android app still connects over the legacy TCP protocol.

## 2. Backwards compatibility with the Freenove Android app

The legacy TCP protocol on ports 5003 (commands) and 8003 (video) is preserved exactly as Freenove designed it. The same message format (`CMD_NAME#param1#param2\n`) and video framing (4-byte LE length prefix + JPEG) are used.

**Why:** The Android app is useful for quick testing without spinning up the full stack. It also serves as a reference implementation when debugging protocol issues.

## 3. One source of truth for hardware constants

All motor limits, servo ranges, servo defaults, and protocol constants live in `shared/protocol.py`. Drivers import from there. The web dashboard reads the same values via the WebSocket API.

**Why:** Servo limits were tuned iteratively (grabber was 150°, then 140°; arm min was 90°, then 60°, then 75°). Having a single place to change them prevents the web UI and robot from disagreeing.

## 4. Protect the hardware from software

- **Motor ramping** (400 duty/step, 50 Hz): prevents current spikes that brown out the Pi.
- **Servo sweeping** (3°/step, 50 Hz): prevents sudden jerky motion and reduces current draw.
- **Clamping everywhere**: all motor duty and servo angle values are clamped to safe ranges before reaching hardware. Out-of-range values are silently corrected.
- **No servo release on shutdown**: `servo.close()` is a no-op. Releasing PWM causes the arm to drop and the grabber to fly open. `os._exit(0)` handles cleanup.

## 5. LiveView with zero DOM patches on the Blockly page

Blockly creates SVG elements with IDs like `blockly-1` in both the workspace and flyout. Phoenix LiveView's DOM differ detects these as duplicate IDs and logs warnings on every patch. The solution: `blocks_live.ex` has **no server-side assign changes** during operation. All dynamic UI (distance display, run/stop toggle, log panel) is handled purely in JavaScript via `push_event`.

The dashboard page uses normal LiveView assigns — the Blockly-specific constraint only applies to `blocks_live.ex`.

## 6. Client-side Blockly execution

Blockly programs are compiled to JavaScript and executed in the browser, not on the server. The generated code calls an async `robot` API that pushes LiveView events (`motor`, `stop`, `servo`, `led`, etc.) back to the server.

**Why:** Server-side execution would require round-trips for every block, making loops and conditionals sluggish. Client-side execution gives instant feedback and supports `await`-based timing (wait blocks, drive-for-N-seconds).

**Trade-off:** The browser must stay connected. If the tab closes mid-program, the robot keeps doing whatever the last command was (mitigated by the stop event on program end/error).

## 7. Async all the way down

The robot server is a single Python asyncio process. Hardware calls that block (ultrasonic timing, camera capture) are wrapped in `run_in_executor`. WebSocket servers, TCP servers, video streaming, telemetry broadcast, and autonomous behaviors all run as concurrent tasks.

**Why:** Multiple threads with shared GPIO state is a recipe for race conditions. A single event loop with explicit concurrency (tasks, executors) is easier to reason about and debug.

## 8. Minimal dependencies

- **Pi side**: `gpiozero`, `picamera2`, `rpi-hardware-pwm`, `websockets`. No web framework, no ORM, no config library.
- **Web side**: Phoenix + LiveView + esbuild + Tailwind. One npm package (`blockly`).
- **Vision side**: `ultralytics` (bundles YOLO + torch + OpenCV).

**Why:** Fewer dependencies = fewer things to break on a constrained device. The Pi doesn't even need Node.js or npm.

## 9. Task automation over documentation

Deployment, setup, and development workflows are encoded in `Taskfile.yml`. A single `task dev` syncs code and restarts the robot. A single `task setup` bootstraps everything from scratch.

**Why:** Setup instructions go stale. Executable task definitions don't. If a step changes, the Taskfile changes with it.

## 10. Degrade gracefully

- No event loop running? Servo/motor commands apply immediately instead of ramping.
- No WebSocket clients? Video loop sleeps instead of encoding frames.
- No desktop GPU? Vision engine simply doesn't start — robot runs fine without it.
- Pi version 4 instead of 5? Config auto-detects and picks the right driver backend.
- PCB v1.0 instead of v2.0? Different servo/LED drivers load automatically.
