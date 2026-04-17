"""MPU-6050 6-axis IMU (GY-521 breakout) over I2C bus 1.

Default config:
  accel: +/-2g    (16384 LSB/g)
  gyro:  +/-250   dps (131 LSB/dps)
  DLPF:  ~42 Hz   low-pass (register 0x1A = 3)

Reads are cheap (~1 ms burst read); safe to poll at 50-100 Hz for
feeding a gyro-Z yaw integrator into the scan-match rotation prior.

Calibration
-----------
The part has non-trivial zero-rate bias (datasheet allows ±20 dps on
gyro) and ~±5% accel full-scale tolerance. Two corrections are applied:

1. **Startup bias**: sample ~2 s at rest, take the mean for each axis.
   Subtracted from gyro readings; accel X/Y get the same treatment so
   they read ~0 when flat. Accel Z is scaled so that it reads exactly
   1 g at rest (orientation permitting — see below).
2. **Adaptive bias**: once calibrated, when the caller tells us we're
   stationary (`mark_stationary()`), we slowly blend the live gyro mean
   into the bias to track thermal drift.

Orientation: on the tank, the board sits upright with +Z pointing
roughly up. At rest accel should therefore read roughly (0, 0, +1g).
If the board is mounted sideways or inverted, the Z-scale correction
will be wrong — but gyro integration for yaw only needs gx/gy/gz
biases, which are axis-agnostic.

Degrades gracefully: if smbus2 is missing or the chip doesn't ACK,
`available` stays False and `read()` returns None.
"""

from __future__ import annotations

import contextlib
import logging
import time

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

# Calibration defaults
_CALIB_DURATION_S = 2.0
_CALIB_RATE_HZ = 100.0
# Adaptive update blend when stationary: bias += alpha * (live - bias)
_ADAPTIVE_ALPHA = 0.01


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
    def __init__(self, *, calibrate: bool = True) -> None:
        self._bus = None
        self._available = False

        # Calibration offsets (subtracted from raw). None while uncalibrated.
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._accel_bias_xy = [0.0, 0.0]
        self._accel_z_scale = 1.0
        self._calibrated = False

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
            return

        if calibrate:
            self.calibrate()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, duration_s: float = _CALIB_DURATION_S, rate_hz: float = _CALIB_RATE_HZ) -> bool:
        """Block for ~duration_s sampling at rest, capture gyro/accel bias.

        Caller must keep the robot perfectly still during this window.
        Returns True on success.
        """
        if not self._available:
            return False

        period = 1.0 / rate_hz
        n_target = max(10, int(duration_s * rate_hz))

        gx_sum = gy_sum = gz_sum = 0.0
        ax_sum = ay_sum = az_sum = 0.0
        n = 0
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and n < n_target * 2:
            r = self._raw_read()
            if r is not None:
                ax, ay, az, gx, gy, gz, _ = r
                ax_sum += ax
                ay_sum += ay
                az_sum += az
                gx_sum += gx
                gy_sum += gy
                gz_sum += gz
                n += 1
            time.sleep(period)

        if n < 10:
            log.warning("IMU calibration failed: only %d samples", n)
            return False

        self._gyro_bias = [gx_sum / n, gy_sum / n, gz_sum / n]
        self._accel_bias_xy = [ax_sum / n, ay_sum / n]
        az_mean = az_sum / n
        # Assume +Z is up at rest; compensate scale so az reads exactly 1 g.
        self._accel_z_scale = (1.0 / az_mean) if abs(az_mean) > 0.1 else 1.0
        self._calibrated = True
        log.info(
            "IMU calibrated (n=%d): gyro_bias=(%.2f,%.2f,%.2f) dps, accel_xy_bias=(%.3f,%.3f) g, az_scale=%.4f",
            n,
            self._gyro_bias[0],
            self._gyro_bias[1],
            self._gyro_bias[2],
            self._accel_bias_xy[0],
            self._accel_bias_xy[1],
            self._accel_z_scale,
        )
        return True

    def mark_stationary(self, reading: ImuReading) -> None:
        """Caller signals the robot is stationary; blend this reading into bias.

        Use this when motors are idle and measured gyro is tiny — lets the
        bias track thermal drift without interrupting normal operation.
        """
        if not self._calibrated:
            return
        # The passed-in reading is already bias-corrected, so it represents
        # residual error. Blend a small fraction back into the bias.
        self._gyro_bias[0] += _ADAPTIVE_ALPHA * reading.gx
        self._gyro_bias[1] += _ADAPTIVE_ALPHA * reading.gy
        self._gyro_bias[2] += _ADAPTIVE_ALPHA * reading.gz

    def _raw_read(self) -> tuple[float, float, float, float, float, float, float] | None:
        """Return raw (ax, ay, az, gx, gy, gz, temp_c) without calibration."""
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
        return ax, ay, az, gx, gy, gz, temp

    def read(self) -> ImuReading | None:
        r = self._raw_read()
        if r is None:
            return None
        ax, ay, az, gx, gy, gz, temp = r
        return ImuReading(
            ax=ax - self._accel_bias_xy[0],
            ay=ay - self._accel_bias_xy[1],
            az=az * self._accel_z_scale,
            gx=gx - self._gyro_bias[0],
            gy=gy - self._gyro_bias[1],
            gz=gz - self._gyro_bias[2],
            temp_c=temp,
        )

    def close(self) -> None:
        if self._bus is not None:
            with contextlib.suppress(Exception):
                self._bus.close()
            self._bus = None
        self._available = False
