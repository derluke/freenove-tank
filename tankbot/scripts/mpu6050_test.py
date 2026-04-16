"""Quick MPU-6050 sanity test.

Run on the Pi:
    python3 scripts/mpu6050_test.py

Prints WHO_AM_I, then streams gyro (deg/s) + accel (g) + temp (C).
Ctrl-C to stop.
"""

from __future__ import annotations

import time

from smbus2 import SMBus

ADDR = 0x68
BUS = 1

# Registers
PWR_MGMT_1 = 0x6B
WHO_AM_I = 0x75
ACCEL_XOUT_H = 0x3B
TEMP_OUT_H = 0x41
GYRO_XOUT_H = 0x43

# Default sensitivity (no config): +/-2g, +/-250 deg/s
ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0


def read_s16(bus: SMBus, reg: int) -> int:
    hi = bus.read_byte_data(ADDR, reg)
    lo = bus.read_byte_data(ADDR, reg + 1)
    v = (hi << 8) | lo
    if v & 0x8000:
        v -= 1 << 16
    return v


def main() -> None:
    bus = SMBus(BUS)

    who = bus.read_byte_data(ADDR, WHO_AM_I)
    print(f"WHO_AM_I = 0x{who:02x} (expect 0x68)")

    # Wake up (clear SLEEP bit).
    bus.write_byte_data(ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)

    print("Streaming. Tilt / rotate the board. Ctrl-C to stop.")
    try:
        while True:
            ax = read_s16(bus, ACCEL_XOUT_H) / ACCEL_LSB_PER_G
            ay = read_s16(bus, ACCEL_XOUT_H + 2) / ACCEL_LSB_PER_G
            az = read_s16(bus, ACCEL_XOUT_H + 4) / ACCEL_LSB_PER_G
            temp_raw = read_s16(bus, TEMP_OUT_H)
            temp_c = temp_raw / 340.0 + 36.53
            gx = read_s16(bus, GYRO_XOUT_H) / GYRO_LSB_PER_DPS
            gy = read_s16(bus, GYRO_XOUT_H + 2) / GYRO_LSB_PER_DPS
            gz = read_s16(bus, GYRO_XOUT_H + 4) / GYRO_LSB_PER_DPS
            print(
                f"acc=({ax:+.2f},{ay:+.2f},{az:+.2f})g  "
                f"gyro=({gx:+7.1f},{gy:+7.1f},{gz:+7.1f})dps  "
                f"T={temp_c:.1f}C",
                end="\r",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
