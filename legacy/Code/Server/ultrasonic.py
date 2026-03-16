import time
from parameter import ParameterManager

class gpiozero_ultrasonic:
    def __init__(self, trigger_pin=27, echo_pin=22):
        try:
            from gpiozero import DistanceSensor, PWMSoftwareFallback
            import warnings
            
            # Initialize the ultrasonic sensor using gpiozero
            warnings.filterwarnings("ignore", category=PWMSoftwareFallback)  # Ignore PWM software fallback warnings
            self.trigger_pin = trigger_pin  # Set the trigger pin number
            self.echo_pin = echo_pin     # Set the echo pin number
            self.sensor = DistanceSensor(echo=self.echo_pin, trigger=self.trigger_pin, max_distance=3)  # Initialize the distance sensor
        except ImportError:
            raise RuntimeError("gpiozero library not available")

    def get_distance(self):
        """Get the distance measurement from the ultrasonic sensor in centimeters."""
        try:
            distance_cm = self.sensor.distance * 100  # Convert distance from meters to centimeters
            return round(float(distance_cm), 1)       # Return the distance rounded to one decimal place
        except Exception:
            return -1  # Return error code on failure

    def close(self):
        """Close the distance sensor."""
        if hasattr(self, 'sensor'):
            self.sensor.close()        # Close the sensor to release resources


class lgpiod_ultrasonic:
    def __init__(self, trigger_pin=27, echo_pin=22):
        try:
            import lgpio
            
            self.lgpio = lgpio  # Cache the module reference
            self.trigger_pin = trigger_pin
            self.echo_pin = echo_pin
            try:
                self.chip = lgpio.gpiochip_open(0)
            except:
                self.chip = lgpio.gpiochip_open(4)
            lgpio.gpio_claim_output(self.chip, self.trigger_pin)
            lgpio.gpio_claim_input(self.chip, self.echo_pin)
        except ImportError:
            raise RuntimeError("lgpio library not available")

    def get_distance(self):
        """Get the distance measurement from the ultrasonic sensor in centimeters."""
        try:
            lgpio = self.lgpio  # Use cached module reference
            
            lgpio.gpio_write(self.chip, self.trigger_pin, 0)
            time.sleep(0.05)

            lgpio.gpio_write(self.chip, self.trigger_pin, 1)
            time.sleep(0.00001)  # 10us
            lgpio.gpio_write(self.chip, self.trigger_pin, 0)

            timeout = time.time() + 1.0
            start_time = time.time()
            
            # Wait for echo to go high
            while lgpio.gpio_read(self.chip, self.echo_pin) == 0:
                start_time = time.time()
                if start_time > timeout:
                    return -1  # Error code for timeout

            # Wait for echo to go low
            stop_time = time.time()
            while lgpio.gpio_read(self.chip, self.echo_pin) == 1:
                stop_time = time.time()
                if stop_time > timeout:
                    return -1  # Error code for timeout

            duration = stop_time - start_time
            distance = (duration * 34300) / 2
            return round(float(distance), 1)
        except Exception:
            return -1  # Return error code on failure

    def close(self):
        """Close the lgpio chip connection."""
        if hasattr(self, 'chip') and self.chip is not None:
            try:
                self.lgpio.gpiochip_close(self.chip)
                self.chip = None
            except Exception:
                pass  # Ignore errors during cleanup


class Ultrasonic:
    def __init__(self, trigger_pin=27, echo_pin=22):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.param_manager = ParameterManager()
        self.pi_version = self.param_manager.get_pi_version()
        self.sensor = None  # Initialize as None for proper cleanup
        
        if self.pi_version == 2:  # Raspberry Pi 5
            print("Using lgpiod_ultrasonic")
            self.sensor = lgpiod_ultrasonic(trigger_pin, echo_pin)
        else:  # Raspberry Pi 4 or earlier
            print("Using gpiozero_ultrasonic")
            self.sensor = gpiozero_ultrasonic(trigger_pin, echo_pin)

    def get_distance(self):
        """Get the distance measurement from the ultrasonic sensor."""
        if self.sensor is None:
            return -1
        return self.sensor.get_distance()

    def close(self):
        """Close the ultrasonic sensor and release resources."""
        if self.sensor is not None:
            self.sensor.close()
            self.sensor = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.close()

if __name__ == '__main__':
    with Ultrasonic() as ultrasonic:
        try:
            while True:
                distance = ultrasonic.get_distance()
                if distance >= 0:
                    print(f"Ultrasonic distance: {distance}cm")
                else:
                    print("Distance measurement failed")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nEnd of program")