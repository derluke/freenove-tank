"""Robot entry point — ties together drivers, protocol servers, and the control loop."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import signal
import socket
import struct

from tankbot.robot.config import RobotConfig
from tankbot.robot.drivers import Camera, InfraredSensors, LedStrip, Motor, ServoController, Ultrasonic
from tankbot.robot.protocol.legacy_tcp import LegacyCmdServer, LegacyVideoServer
from tankbot.robot.protocol.websocket_api import WebSocketAPI
from tankbot.shared.protocol import CarMode, Cmd, LedEffect, Message

log = logging.getLogger(__name__)


def _get_ip() -> str:
    """Get wlan0 IP address, falling back to hostname lookup."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", b"wlan0"[:15]))[20:24])
    except Exception:
        return socket.gethostbyname(socket.gethostname())


class Robot:
    def __init__(self, config: RobotConfig | None = None) -> None:
        self.cfg = config or RobotConfig.load()
        self.ip = _get_ip()

        # Hardware
        self.motor = Motor()
        self.servo = ServoController(self.cfg.pcb_version, self.cfg.pi_version)
        self.ultrasonic = Ultrasonic(pi_version=self.cfg.pi_version)
        self.infrared = InfraredSensors(self.cfg.pcb_version)
        self.led = LedStrip(self.cfg.pcb_version, self.cfg.pi_version)
        self.camera = Camera(stream_size=self.cfg.stream_size, hflip=True, vflip=True)

        # State
        self.car_mode = CarMode.MANUAL
        self.left_speed = 0
        self.right_speed = 0
        self.vision_status: dict | None = None  # latest from vision engine

        # Servers
        self.legacy_cmd = LegacyCmdServer(on_message=self._handle_legacy_cmd)
        self.legacy_video = LegacyVideoServer()
        self.ws_api = WebSocketAPI(on_command=self._handle_ws_cmd)

        self._running = False

    # ------------------------------------------------------------------
    # Legacy protocol handler (Freenove Android app)
    # ------------------------------------------------------------------
    def _handle_legacy_cmd(self, raw: str) -> None:
        msg = Message.parse(raw)
        p = msg.int_params

        if msg.command == Cmd.MOTOR and len(p) >= 2:
            self.left_speed, self.right_speed = p[0], p[1]
            self.motor.set(self.left_speed, self.right_speed)

        elif msg.command == Cmd.SERVO and len(p) >= 2:
            if self.car_mode in (CarMode.MANUAL, CarMode.SONAR):
                self.servo.set_angle(p[0], p[1])

        elif msg.command == Cmd.LED and len(p) >= 5:
            self._handle_led(p)

        elif msg.command == Cmd.MODE and len(p) >= 1:
            self._handle_mode_switch(p[0])

        elif msg.command == Cmd.ACTION and len(p) >= 1:
            self._handle_action(p[0])

    def _handle_led(self, params: list[int]) -> None:
        effect = params[0]
        if effect == LedEffect.STATIC:
            self.led.set_by_mask(params[4], params[1], params[2], params[3])
        elif effect == LedEffect.OFF:
            self.led.off()
        # More complex effects (wipe, breathe, rainbow) run in the LED task

    def _handle_mode_switch(self, mode: int) -> None:
        if mode == 0:
            self.car_mode = CarMode.MANUAL
            self.motor.stop()
        elif mode == 1:
            self.car_mode = CarMode.SONAR
        elif mode == 2:
            self.car_mode = CarMode.INFRARED

    def _handle_action(self, action: int) -> None:
        if action == 0:
            self.car_mode = CarMode.CLAMP_STOP
        elif action == 1:
            self.car_mode = CarMode.CLAMP_UP
        elif action == 2:
            self.car_mode = CarMode.CLAMP_DOWN

    # ------------------------------------------------------------------
    # WebSocket protocol handler (Phoenix app, autonomy, Blockly)
    # ------------------------------------------------------------------
    async def _handle_ws_cmd(self, msg: dict) -> None:
        cmd_type = msg.get("cmd")
        if cmd_type == "motor":
            left, right = int(msg.get("left", 0)), int(msg.get("right", 0))
            self.left_speed, self.right_speed = left, right
            self.motor.set(left, right)
        elif cmd_type == "servo":
            self.servo.set_angle(int(msg.get("channel", 0)), int(msg.get("angle", 90)))
        elif cmd_type == "led":
            self.led.set_by_mask(
                int(msg.get("mask", 0xF)), int(msg.get("r", 0)), int(msg.get("g", 0)), int(msg.get("b", 0))
            )
        elif cmd_type == "led_off":
            self.led.off()
        elif cmd_type == "mode":
            self._handle_mode_switch(int(msg.get("mode", 0)))
        elif cmd_type == "stop":
            self.motor.stop()
        elif cmd_type == "vision":
            # Store and re-broadcast vision engine status to all clients
            self.vision_status = {
                "type": "vision",
                "state": msg.get("state", "unknown"),
                "detections": msg.get("detections", []),
                "distance": msg.get("distance", 0),
            }
            if "depth" in msg:
                self.vision_status["depth"] = msg["depth"]
            if "slam" in msg:
                self.vision_status["slam"] = msg["slam"]
            if "autonomy" in msg:
                self.vision_status["autonomy"] = msg["autonomy"]
            if self.ws_api.has_clients:
                await self.ws_api.broadcast_telemetry(self.vision_status)
        elif cmd_type == "arm":
            # Ch1 = arm (90° down, 150° up)
            direction = msg.get("dir")
            step = int(msg.get("step", 5))
            if direction == "up":
                new = self.servo.get_target(1) + step
            elif direction == "down":
                new = self.servo.get_target(1) - step
            else:
                new = int(msg.get("angle", self.servo.get_target(1)))
            self.servo.set_angle(1, new)
        elif cmd_type == "grabber":
            # Ch0 = grabber (90° open, 150° closed)
            direction = msg.get("dir")
            step = int(msg.get("step", 5))
            if direction == "open":
                new = self.servo.get_target(0) - step
            elif direction == "close":
                new = self.servo.get_target(0) + step
            else:
                new = int(msg.get("angle", self.servo.get_target(0)))
            self.servo.set_angle(0, new)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------
    async def _video_loop(self) -> None:
        self.camera.start_stream()
        try:
            while self._running:
                has_any = self.legacy_video.has_clients or self.ws_api.has_video_clients
                if not has_any:
                    await asyncio.sleep(0.1)
                    continue

                frame = await self.camera.get_frame()
                if frame is None:
                    continue
                tasks = []
                if self.legacy_video.has_clients:
                    tasks.append(self.legacy_video.send_frame(frame))
                if self.ws_api.has_video_clients:
                    tasks.append(self.ws_api.broadcast_frame(frame))
                if tasks:
                    await asyncio.gather(*tasks)
        finally:
            self.camera.stop_stream()

    async def _telemetry_loop(self) -> None:
        while self._running:
            distance = await asyncio.get_running_loop().run_in_executor(None, self.ultrasonic.get_distance)
            # Send to legacy clients
            if self.legacy_cmd.has_clients:
                await self.legacy_cmd.send_to_all(f"CMD_SONIC#{distance:.2f}")
            # Send to WS clients
            if self.ws_api.has_clients:
                await self.ws_api.broadcast_telemetry(
                    {
                        "type": "telemetry",
                        "distance": distance,
                        "mode": self.car_mode.value,
                        "motor": {"left": self.left_speed, "right": self.right_speed},
                        "ir": await asyncio.get_running_loop().run_in_executor(None, self.infrared.read),
                    }
                )
            await asyncio.sleep(1.0)

    async def _car_behavior_loop(self) -> None:
        """Runs built-in autonomous behaviors (sonar avoidance, line following)."""
        while self._running:
            if self.car_mode == CarMode.SONAR:
                distance = await asyncio.get_running_loop().run_in_executor(None, self.ultrasonic.get_distance)
                if distance > 0 and distance < 45:
                    self.motor.set(-1500, -1500)
                    await asyncio.sleep(0.4)
                    self.motor.set(-1500, 1500)
                    await asyncio.sleep(0.2)
                else:
                    self.motor.set(1500, 1500)
                await asyncio.sleep(0.2)

            elif self.car_mode == CarMode.INFRARED:
                ir = await asyncio.get_running_loop().run_in_executor(None, self.infrared.read)
                speed_map = {
                    2: (1200, 1200),
                    4: (-1500, 2500),
                    6: (-2000, 4000),
                    1: (2500, -1500),
                    3: (4000, -2000),
                    7: (0, 0),
                }
                if ir in speed_map:
                    self.motor.set(*speed_map[ir])

            else:
                await asyncio.sleep(0.05)
                continue

            await asyncio.sleep(0.02)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._running = True
        log.info("Starting robot on %s", self.ip)

        await self.legacy_cmd.start(self.ip, self.cfg.cmd_port)
        await self.legacy_video.start(self.ip, self.cfg.video_port)
        await self.ws_api.start(self.ip, self.cfg.ws_port)

        log.info(
            "Servers running — cmd:%d  video:%d  ws:%d",
            self.cfg.cmd_port,
            self.cfg.video_port,
            self.cfg.ws_port,
        )

        tasks = [
            asyncio.create_task(self._video_loop()),
            asyncio.create_task(self._telemetry_loop()),
            asyncio.create_task(self._car_behavior_loop()),
        ]

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        await stop.wait()
        log.info("Shutting down...")

        self._running = False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.shutdown()

        # Force exit — camera thread can block asyncio from exiting
        import os

        os._exit(0)

    async def shutdown(self) -> None:
        self.motor.stop_immediate()
        self.led.off()
        await self.legacy_cmd.stop()
        await self.legacy_video.stop()
        await self.ws_api.stop()
        self.camera.close()
        self.servo.close()
        self.ultrasonic.close()
        self.infrared.close()
        log.info("Robot shut down cleanly")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    robot = Robot()
    asyncio.run(robot.run())


if __name__ == "__main__":
    main()
