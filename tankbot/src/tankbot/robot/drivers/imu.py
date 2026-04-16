"""MPU-6050 6-axis IMU (GY-521 breakout) over I2C bus 1.

Default config:
  accel: +/-2g    (16384 LSB/g)
  gyro:  +/-250   dps (131 LSB/dps)
  DLPF:  ~42 Hz   low-pass (register 0x1A = 3)

Reads are cheap (~1 ms burst read); safe to poll at 50-100 Hz later
when we wire gyro-Z into the scan-match rotation prior.

Degrades gracefully: if smbus2 is missing or the chip doesn't ACK,
`available` stays False and `read()` returns None.
"""

from __future__ import annotations

import contextlib
import logging

log = logging.getLogger(__name__)

_ADDR = 0x68
_BUS = 1

# Registers
_PWR_MGMT_1 = 0x6B
_CONFIG = 0x1A
_GYRO_CONFIG = 0x1B
_ACCEL_CONFIG = 0x1C
_WHO_AM_I = 0x75
_ACCEL_XOUT_H = 0x3B  # 14 bytes: ax,ay,az,temp,gx,gy,gz

_ACCEL_LSB_PER_G = 16384.0
_GYRO_LSB_PER_DPS = 131.0


class ImuReading:
    __slots__ = ("ax", "ay", "az", "gx", "gy", "gz", "temp_c")

    def __init__(self, ax: float, ay: float, az: float, gx: float, gy: float, gz: float, temp_c: float) -> None:
        self.ax = ax
        self.ay = ay
        self.az = az
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.temp_c = temp_c

    def as_dict(self) -> dict:
        return {
            "ax": round(self.ax, 3),
            "ay": round(self.ay, 3),
            "az": round(self.az, 3),
            "gx": round(self.gx, 2),
            "gy": round(self.gy, 2),
            "gz": round(self.gz, 2),
            "temp_c": round(self.temp_c, 1),
        }


class Imu:
    def __init__(self) -> None:
        self._bus = None
        self._available = False
        try:
            from smbus2 import SMBus

            self._bus = SMBus(_BUS)
            who = self._bus.read_byte_data(_ADDR, _WHO_AM_I)
            if who != 0x68:
                log.warning("MPU-6050 WHO_AM_I=0x%02x (expected 0x68)", who)
                return
            # Wake (clear SLEEP bit), set clock source to gyro-X PLL for stability.
            self._bus.write_byte_data(_ADDR, _PWR_MGMT_1, 0x01)
            # DLPF ~42 Hz — kills motor noise that rides on the rails.
            self._bus.write_byte_data(_ADDR, _CONFIG, 0x03)
            # Default ranges (±2g, ±250 dps).
            self._bus.write_byte_data(_ADDR, _GYRO_CONFIG, 0x00)
            self._bus.write_byte_data(_ADDR, _ACCEL_CONFIG, 0x00)
            self._available = True
            log.info("IMU ready (MPU-6050 @ 0x68 on bus %d)", _BUS)
        except Exception as exc:
            log.warning("IMU unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def read(self) -> ImuReading | None:
        if not self._available or self._bus is None:
            return None
        try:
            data = self._bus.read_i2c_block_data(_ADDR, _ACCEL_XOUT_H, 14)
        except Exception as exc:
            log.debug("IMU read failed: %s", exc)
            return None

        def s16(hi: int, lo: int) -> int:
            v = (hi << 8) | lo
            return v - (1 << 16) if v & 0x8000 else v

        ax = s16(data[0], data[1]) / _ACCEL_LSB_PER_G
        ay = s16(data[2], data[3]) / _ACCEL_LSB_PER_G
        az = s16(data[4], data[5]) / _ACCEL_LSB_PER_G
        temp = s16(data[6], data[7]) / 340.0 + 36.53
        gx = s16(data[8], data[9]) / _GYRO_LSB_PER_DPS
        gy = s16(data[10], data[11]) / _GYRO_LSB_PER_DPS
        gz = s16(data[12], data[13]) / _GYRO_LSB_PER_DPS
        return ImuReading(ax, ay, az, gx, gy, gz, temp)

    def close(self) -> None:
        if self._bus is not None:
            with contextlib.suppress(Exception):
                self._bus.close()
            self._bus = None
        self._available = False
