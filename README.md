# Tankbot

A modern control system for the [Freenove Tank Robot Kit](https://github.com/Freenove/Freenove_Tank_Robot_Kit_for_Raspberry_Pi) on Raspberry Pi 5.

Replaces the original PyQt5 desktop app with a three-part architecture:

| Component | Runs on | Language | Purpose |
|-----------|---------|----------|---------|
| **Robot server** | Raspberry Pi | Python 3.11 (asyncio) | Hardware drivers, protocol servers |
| **Web dashboard** | Desktop / any device | Elixir (Phoenix LiveView) | Real-time control UI, Blockly programming |
| **Vision engine** | Desktop GPU | Python (YOLOv11) | Autonomous obstacle avoidance |

## Quick start

### Prerequisites

- Raspberry Pi 5 with the Freenove tank kit assembled (PCB v2.0)
- Desktop machine on the same network
- [Task](https://taskfile.dev) runner, [uv](https://docs.astral.sh/uv/) (Python), [Elixir](https://elixir-lang.org/install.html) + [Node.js](https://nodejs.org/)

### Configuration

```bash
cd tankbot
cp .env.example .env
# Edit .env with your Pi's IP, username, and deploy path
```

| Variable | Default | Used by |
|----------|---------|---------|
| `PI_HOST` | `raspberrypi.local` | Taskfile (SSH/rsync) |
| `PI_USER` | `pi` | Taskfile (SSH/rsync) |
| `PI_DIR` | `/home/pi/tankbot` | Taskfile (deploy path) |
| `ROBOT_URL` | `ws://localhost:9000` | Phoenix web app, vision engine |

### First-time setup

```bash
task setup        # installs uv on Pi, deploys code, installs Phoenix deps, installs vision deps
```

### Running

Open three terminals:

```bash
# Terminal 1 — Robot server on Pi
task pi:start

# Terminal 2 — Web dashboard on desktop
task web:start        # → http://localhost:4000

# Terminal 3 — (optional) Vision autonomy
task vision:start
```

### Day-to-day development

```bash
task dev              # sync code to Pi + restart robot server
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (any device)                               │
│  ┌───────────────┐  ┌────────────────────────────┐  │
│  │  Dashboard     │  │  Block Programming         │  │
│  │  WASD drive    │  │  Blockly → JS code gen     │  │
│  │  Arm/grabber   │  │  Save/load programs        │  │
│  │  Camera feed   │  │  Async robot API           │  │
│  │  Telemetry     │  │  Client-side execution     │  │
│  └───────┬───────┘  └──────────┬─────────────────┘  │
│          │ LiveView            │ LiveView             │
└──────────┼─────────────────────┼─────────────────────┘
           ▼                     ▼
┌──────────────────────────────────────┐
│  Phoenix (Elixir)                    │
│  WebSocket bridge to robot           │
│  PubSub: telemetry + video frames    │
└──────────────────┬───────────────────┘
                   │ WebSocket :9000
                   ▼
┌──────────────────────────────────────┐     ┌────────────────────┐
│  Robot server (Python, asyncio)      │◄────│  Vision engine     │
│  ┌──────────┐ ┌────────────────────┐ │     │  YOLOv11 nano      │
│  │ Legacy   │ │ WebSocket API      │ │     │  Desktop GPU       │
│  │ TCP:5003 │ │ JSON + binary      │ │     │  ~10 Hz decisions  │
│  │ TCP:8003 │ │ video frames       │ │     └────────────────────┘
│  └──────────┘ └────────────────────┘ │
│  Drivers: motor, servo, camera,      │
│  ultrasonic, infrared, LED strip     │
└──────────────────────────────────────┘
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

**Keyboard:**
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

**Buttons:** Click or hold — hold-to-repeat at 120ms for continuous servo movement.

### Block programming (http://localhost:4000/blocks)

Visual programming with Google Blockly. Custom robot blocks:
- **Movement**: drive forward/backward, turn left/right, stop
- **Arm & Grabber**: set arm angle, open/half-open/close grabber
- **LEDs**: set color, turn off
- **Sensors**: read ultrasonic distance
- **Timing**: wait N seconds
- **Logging**: print to log panel

Plus standard Blockly blocks: if/else, loops, variables, math, text.

Programs are saved to browser localStorage. Auto-saves every 2 seconds.

## Hardware

### Servo channels

| Channel | GPIO | Function | Range | Default |
|---------|------|----------|-------|---------|
| 0 | 7 | Grabber | 90° (open) – 140° (closed) | 130° |
| 1 | 8 | Arm | 75° (down) – 150° (up) | 140° |
| 2 | 25 | Camera tilt | 0° – 180° | — |

### Motor ramping

Motors ramp at 400 duty/step (50 Hz) to prevent power spikes that cause Pi undervoltage. Emergency stop bypasses ramping.

### Servo sweeping

Servos move at 3°/step (50 Hz) for smooth motion. On shutdown, servos hold position (no release) to prevent the arm from dropping.

### PCB versions

This project defaults to **PCB v2.0** with hardware PWM servos and SPI LEDs. PCB v1.0 (gpiozero servos, GPIO LEDs) is also supported — set `Pcb_Version` in `params.json` on the Pi.

See [legacy/README.md](legacy/) for the original Freenove GPIO pinout table.

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
│       ├── vision.py                 YOLOv11 autonomous navigation
│       └── robot_client.py           WebSocket client for vision
├── tankbot_web/
│   ├── mix.exs                       Elixir deps (Phoenix, LiveView)
│   ├── lib/tankbot_web/
│   │   ├── robot_socket.ex           WebSocket bridge to Python
│   │   └── ...
│   ├── lib/tankbot_web_web/live/
│   │   ├── dashboard_live.ex         Main control UI
│   │   └── blocks_live.ex            Blockly visual programming
│   └── assets/js/hooks/
│       ├── drive_hook.js             WASD + hold-to-repeat controls
│       └── blockly_hook.js           Blockly workspace + code gen
└── ...

legacy/                               Original Freenove codebase (reference)
├── Code/                             PyQt5 server + client
├── Application/                      Pre-built desktop clients
├── Tutorial.pdf                      Assembly & wiring guide
└── ...
```

## Task commands

```bash
# Deployment
task pi:sync          # rsync code to Pi
task pi:deploy        # sync + install deps
task pi:start         # start robot server (SSH)
task pi:stop          # stop robot server
task pi:restart       # stop + deploy + start
task pi:ssh           # open SSH session

# Web dashboard
task web:setup        # install deps + build assets
task web:start        # start Phoenix dev server

# Vision
task vision:setup     # install YOLO + torch
task vision:start     # start autonomous vision

# Quality
task lint             # ruff linter
task typecheck        # mypy strict
task check            # all checks

# Workflows
task setup            # first-time setup (everything)
task dev              # quick iteration (sync + restart Pi)
task up               # deploy + start robot
```

## License

Original Freenove code: [CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/).
New tankbot code: same license unless otherwise noted.
