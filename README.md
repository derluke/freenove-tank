# Tankbot

A modern control system for the [Freenove Tank Robot Kit](https://github.com/Freenove/Freenove_Tank_Robot_Kit_for_Raspberry_Pi) on Raspberry Pi 5.

Replaces the original PyQt5 desktop app with a three-part architecture:

| Component | Runs on | Language | Purpose |
|-----------|---------|----------|---------|
| **Robot server** | Raspberry Pi 5 | Python 3.11 (asyncio) | Hardware drivers, protocol servers |
| **Web dashboard** | Desktop / any device | Elixir (Phoenix LiveView) | Real-time control UI, Blockly programming |
| **Vision engine** | Desktop GPU | Python (YOLO26 + Depth Pro) | Autonomous navigation with metric depth |

## Prerequisites

### Hardware

- **Freenove Tank Robot Kit** assembled on a **Raspberry Pi 5** (PCB v2.0)
- **Desktop/laptop** with an **NVIDIA GPU** (for the vision engine — any modern GPU with 4GB+ VRAM works, the models use ~2GB total)
- Both machines on the **same local network** (Wi-Fi or Ethernet)

### Raspberry Pi setup

1. Flash **Raspberry Pi OS (Bookworm, 64-bit)** onto the Pi
2. Enable **SSH** and **camera** in `raspi-config`
3. Set up **SSH key access** from your desktop so you can `ssh pi@<pi-ip>` without a password:
   ```bash
   ssh-copy-id <your-pi-user>@<your-pi-ip>
   ```
4. Note the Pi's IP address (`hostname -I` on the Pi) — you'll need it for the `.env` file

### Desktop setup

Install these tools:

| Tool | What for | Install |
|------|----------|---------|
| [Task](https://taskfile.dev) | Build/deploy runner | `brew install go-task` or see [install docs](https://taskfile.dev/installation/) |
| [uv](https://docs.astral.sh/uv/) | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Elixir](https://elixir-lang.org/install.html) | Phoenix web framework | `brew install elixir` or see install docs |
| [Node.js](https://nodejs.org/) | Phoenix asset pipeline | `brew install node` or use your distro's package |
| NVIDIA drivers + CUDA | GPU inference | See [NVIDIA docs](https://developer.nvidia.com/cuda-downloads) |

## Getting started

All commands run from the `tankbot/` directory.

### 1. Configure

```bash
cd tankbot
cp .env.example .env
```

Edit `.env` with your Pi's details:

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_HOST` | `raspberrypi.local` | Pi hostname or IP address |
| `PI_USER` | `pi` | SSH username on the Pi |
| `PI_DIR` | `/home/pi/tankbot` | Where code gets deployed on the Pi |
| `ROBOT_URL` | `ws://localhost:9000` | Robot WebSocket URL (Phoenix and vision use this) |

### 2. First-time setup

```bash
task setup
```

This single command:
- Installs `uv` on the Pi
- Deploys code to the Pi and installs Python dependencies
- Installs Phoenix (Elixir) dependencies and builds JS assets
- Installs vision engine dependencies (PyTorch, YOLO26, Depth Pro, etc.)

**First run will download model weights** (~400MB for Depth Pro, ~50MB for YOLO26l). This only happens once.

### 3. Running

Open **three terminals**:

```bash
# Terminal 1 — Robot server (runs on the Pi via SSH)
task pi:start

# Terminal 2 — Web dashboard (runs locally)
task web:start
# → open http://localhost:4000

# Terminal 3 — Vision engine (runs locally, uses your GPU)
task vision:start          # autonomous mode
# OR
task vision:debug          # debug mode — see what AI sees while driving manually
```

You should see:
- **Camera feed** in the dashboard with live YOLO bounding boxes (class + distance in meters)
- **Depth proximity bars** (L/C/R) showing nearest obstacle distance in each direction
- **Exploration minimap** tracking where the robot has been and where obstacles are
- **Vision state badge** showing what the robot is doing (Cruising / Avoiding / Backing up)

### 4. Day-to-day development

```bash
task dev                   # sync code to Pi + restart robot server
task pi:restart            # full redeploy + restart
```

## How it works

### Vision engine

The vision engine runs on your desktop GPU and processes the robot's camera feed in real-time:

1. **YOLO26l** detects and classifies objects (people, chairs, etc.)
2. **Apple Depth Pro** estimates metric depth for every pixel — outputting real distances in meters
3. The frame is split into three vertical strips (left/center/right) and the **nearest obstacle distance** in each strip is computed
4. The robot **steers toward the most open space** — speed scales with clearance, steering biases toward whichever direction has the most room
5. An **exploration map** tracks visited areas and obstacle locations via dead-reckoning

The vision engine sends all detection and depth data to the robot, which relays it to the web dashboard for visualization.

### Debug mode

`task vision:debug` runs the full vision pipeline (YOLO + Depth Pro) but sends **no motor commands**. You drive manually via WASD while watching:

- Bounding boxes with object class, confidence, and distance
- Depth proximity bars showing meters to nearest obstacle in each strip
- What state the robot *would* be in (backing up / avoiding / cruising)
- Exploration map building as you drive

This is invaluable for tuning thresholds and understanding what the models see before enabling autonomous driving.

### Dashboard

The web dashboard at `http://localhost:4000` provides:

- **Live camera feed** with YOLO bounding box overlays
- **WASD keyboard controls** for manual driving (hold keys for continuous movement)
- **Arm/grabber controls** (R/F for arm, T/G for grabber)
- **Sensor telemetry** — ultrasonic distance, IR sensors, motor speeds
- **Vision autonomy panel** — state indicator, depth bars, detected objects, exploration minimap
- **Mode selector** — Manual, Sonar avoidance, Line following
- **Block programming** — visual programming with Google Blockly at `/blocks`

## Architecture

```
+---------------------------------------------------------+
|  Browser (any device)                                   |
|  +----------------+  +------------------------------+   |
|  |  Dashboard     |  |  Block Programming           |   |
|  |  WASD drive    |  |  Blockly visual editor       |   |
|  |  Camera + YOLO |  |  Save/load programs          |   |
|  |  Depth bars    |  |  Client-side JS execution    |   |
|  |  Minimap       |  |                              |   |
|  +-------+--------+  +-------------+----------------+   |
|          | LiveView                 | LiveView           |
+----------+--------------------------+-------------------+
           v                          v
+----------------------------------------------+
|  Phoenix (Elixir)                            |
|  WebSocket bridge to robot                   |
|  PubSub: telemetry + video + vision overlays |
+---------------------+------------------------+
                      | WebSocket :9000
                      v
+----------------------------------------------+    +---------------------------+
|  Robot server (Python, asyncio)              |<---|  Vision engine (desktop)  |
|  +----------+ +----------------------------+ |    |  YOLO26l detection        |
|  | Legacy   | | WebSocket API              | |    |  Depth Pro metric depth   |
|  | TCP:5003 | | JSON + binary JPEG         | |    |  Exploration mapping      |
|  | TCP:8003 | | Vision telemetry relay     | |    |  ~3 Hz (detection+depth)  |
|  +----------+ +----------------------------+ |    +---------------------------+
|  Drivers: motor, servo, camera, ultrasonic,  |
|  infrared, LED strip                         |
+----------------------------------------------+
```

### Communication protocols

| Protocol | Port | Format | Used by |
|----------|------|--------|---------|
| Legacy TCP (cmd) | 5003 | `CMD_NAME#param1#param2\n` | Freenove Android app |
| Legacy TCP (video) | 8003 | `[4-byte LE length][JPEG]` | Freenove Android app |
| WebSocket | 9000 | JSON (commands/telemetry) + binary (JPEG) | Web dashboard, Vision engine |

The legacy TCP protocol is maintained for backwards compatibility with the official Freenove Android app.

## Controls

### Dashboard (http://localhost:4000)

| Key | Action |
|-----|--------|
| W / Arrow Up | Drive forward |
| S / Arrow Down | Drive backward |
| A / Arrow Left | Turn left |
| D / Arrow Right | Turn right |
| Space | Emergency stop |
| R | Arm up |
| F | Arm down |
| T | Grabber close |
| G | Grabber open |

All drive keys are additive (W+D = forward-right). Servo keys auto-repeat while held.

### Block programming (http://localhost:4000/blocks)

Visual programming with Google Blockly. Custom robot blocks:
- **Movement**: drive forward/backward, turn left/right, stop
- **Arm & Grabber**: set arm angle, open/half-open/close grabber
- **LEDs**: set color, turn off
- **Sensors**: read ultrasonic distance
- **Timing**: wait N seconds
- **Logging**: print to log panel

Programs are saved to browser localStorage and auto-save every 2 seconds.

## Hardware

### Servo channels

| Channel | GPIO | Function | Range | Default |
|---------|------|----------|-------|---------|
| 0 | 7 | Grabber | 90 (open) - 140 (closed) | 130 |
| 1 | 8 | Arm | 75 (down) - 150 (up) | 140 |
| 2 | 25 | Camera tilt | 0 - 180 | -- |

### Sensors

| Sensor | Type | Pins | Notes |
|--------|------|------|-------|
| Ultrasonic | HC-SR04 | GPIO 27 (trigger), 22 (echo) | Fixed forward-facing, max ~2m, unreliable on soft surfaces |
| Infrared | 3x line sensor | GPIO 16, 26, 21 | Returns 3-bit value (0-7) for line following |
| Camera | Picamera2 | CSI | 400x300 JPEG stream, hflip + vflip |

### Motor ramping

Motors ramp at 400 duty/step at 50 Hz to prevent power spikes. Tank treads need a minimum duty of ~1000 to overcome friction. Emergency stop bypasses ramping.

### PCB versions

This project defaults to **PCB v2.0** with hardware PWM servos and SPI LEDs. PCB v1.0 (gpiozero servos, GPIO LEDs) is also supported.

## Project structure

```
tankbot/
├── pyproject.toml                    Python package config
├── Taskfile.yml                      Build/deploy automation
├── src/tankbot/
│   ├── shared/protocol.py            Message definitions, enums, limits
│   ├── robot/
│   │   ├── main.py                   Robot entry point + control loops
│   │   ├── config.py                 Auto-detect Pi/PCB version
│   │   ├── drivers/                  motor, servo, camera, ultrasonic, infrared, LED
│   │   └── protocol/                 legacy_tcp.py, websocket_api.py
│   └── desktop/autonomy/
│       ├── vision.py                 YOLO26 + Depth Pro autonomous navigation
│       └── robot_client.py           WebSocket client for vision engine
├── tankbot_web/
│   ├── mix.exs                       Elixir deps (Phoenix, LiveView)
│   ├── lib/tankbot_web/
│   │   ├── robot_socket.ex           WebSocket bridge to Python robot
│   │   └── ...
│   ├── lib/tankbot_web_web/live/
│   │   ├── dashboard_live.ex         Main control UI + vision overlays
│   │   └── blocks_live.ex            Blockly visual programming
│   └── assets/js/hooks/
│       ├── drive_hook.js             WASD + hold-to-repeat controls
│       ├── blockly_hook.js           Blockly workspace + code gen
│       └── map_hook.js               Exploration minimap canvas
└── ...

legacy/                               Original Freenove codebase (reference)
├── Code/                             PyQt5 server + client
├── Tutorial.pdf                      Assembly & wiring guide
└── ...
```

## Task commands

```bash
# Deployment
task pi:sync              # rsync code to Pi
task pi:deploy            # sync + install deps
task pi:start             # start robot server (SSH)
task pi:stop              # stop robot server
task pi:restart           # stop + deploy + start
task pi:ssh               # open SSH session

# Web dashboard
task web:setup            # install deps + build assets
task web:start            # start Phoenix dev server

# Vision
task vision:setup         # install YOLO + Depth Pro + torch
task vision:start         # start autonomous vision
task vision:debug         # debug mode — AI overlays, manual driving

# Quality
task lint                 # ruff linter
task lint:fix             # ruff linter with auto-fix
task format               # ruff formatter
task typecheck            # mypy strict
task check                # all checks

# Workflows
task setup                # first-time setup (everything)
task dev                  # quick iteration (sync + restart Pi)
task up                   # deploy + start robot
```

## Troubleshooting

**Robot server won't start (`ModuleNotFoundError: No module named 'tankbot'`)**
The editable install on the Pi may be broken. SSH in and run:
```bash
cd /home/<user>/tankbot && ~/.local/bin/uv pip install -e '.[robot]'
```

**Vision engine can't connect to robot**
Make sure `ROBOT_URL` in `.env` points to the Pi's IP, not `localhost`:
```
ROBOT_URL=ws://192.168.x.x:9000
```

**Motors whine but don't move**
The tank treads need high duty values to overcome friction. If you've changed speed constants, ensure nothing goes below 1000.

**Ultrasonic shows huge distances when close to something**
The HC-SR04 is unreliable on soft surfaces (blankets, cushions). The vision engine filters these readings and relies on Depth Pro as the primary depth sensor.

**Depth Pro downloads on first run**
Model weights (~400MB) are downloaded from HuggingFace on first launch. This only happens once and is cached locally.

## License

Original Freenove code: [CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/).
New tankbot code: same license unless otherwise noted.
