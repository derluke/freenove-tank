"""WS281x LED strip driver (4 LEDs).

PCB v1 + Pi4: Freenove_RPI_WS281X (RGB)
PCB v2:       Freenove_SPI_LedPixel (GRB)
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

LED_COUNT = 4


class LedStrip:
    def __init__(self, pcb_version: int = 2, pi_version: int = 2) -> None:
        self._supported = True
        if pcb_version == 1 and pi_version == 2:
            log.warning("PCB v1 LEDs not supported on Pi 5")
            self._supported = False
            self._strip = None
            return

        if pcb_version == 1:
            from .rpi_ledpixel import Freenove_RPI_WS281X  # type: ignore[import-untyped]
            self._strip = Freenove_RPI_WS281X(LED_COUNT, 255, "RGB")
        else:
            from .spi_ledpixel import Freenove_SPI_LedPixel  # type: ignore[import-untyped]
            self._strip = Freenove_SPI_LedPixel(LED_COUNT, 255, "GRB")

        log.info("LED strip initialised (%d LEDs, pcb=%d)", LED_COUNT, pcb_version)

    @property
    def supported(self) -> bool:
        return self._supported

    def set_pixel(self, index: int, r: int, g: int, b: int) -> None:
        if not self._supported:
            return
        self._strip.set_led_rgb_data(index, (r, g, b))
        self._strip.show()

    def set_by_mask(self, mask: int, r: int, g: int, b: int) -> None:
        """Set LEDs selected by bitmask (bit 0 = LED 0, etc.)."""
        if not self._supported:
            return
        for i in range(LED_COUNT):
            if mask & (1 << i):
                self._strip.set_led_rgb_data(i, (r, g, b))
        self._strip.show()

    def fill(self, r: int, g: int, b: int) -> None:
        if not self._supported:
            return
        for i in range(LED_COUNT):
            self._strip.set_led_rgb_data(i, (r, g, b))
        self._strip.show()

    def off(self) -> None:
        self.fill(0, 0, 0)

    def color_wipe(self, r: int, g: int, b: int, wait_ms: int = 50) -> None:
        if not self._supported:
            return
        for i in range(LED_COUNT):
            self._strip.set_led_rgb_data(i, (r, g, b))
            self._strip.show()
            time.sleep(wait_ms / 1000)

    def close(self) -> None:
        self.off()
