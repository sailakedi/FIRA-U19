import Jetson.GPIO as GPIO
import time

# ——— Configuration ———
SW_PIN = 7               # BOARD pin 7
GPIO.setmode(GPIO.BOARD)
# Input with pull-up: open = HIGH, closed = LOW
GPIO.setup(SW_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize previous state
prev_state = GPIO.input(SW_PIN)

try:
    print(f"Starting: initial state is {'OPEN' if prev_state else 'CLOSED'}")
    while True:
        state = GPIO.input(SW_PIN)
        # Detect change
        if state != prev_state:
            # Report the new and previous states
            if state == GPIO.HIGH:
                print("→ Switch is now OPEN")
            else:
                print("→ Switch is now CLOSED")
            print(f"   (Previous state: {'OPEN' if prev_state else 'CLOSED'})")
            prev_state = state

        time.sleep(0.05)  # 50 ms debounce/check interval

except KeyboardInterrupt:
    print("\nExiting on user interrupt")

finally:
    GPIO.cleanup()
