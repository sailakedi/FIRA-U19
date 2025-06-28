import Jetson.GPIO as GPIO
import time

# Use BOARD numbering; pin 7 is GPIO4 (kernel line 204)
GPIO.setmode(GPIO.BOARD)

# Set up pin 7 as input with the internal pull‑up
GPIO.setup(7, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    print("Press Ctrl‑C to exit.")
    while True:
        if GPIO.input(7) == GPIO.LOW:
            print(" Switch CLOSED (ON)")
        else:
            print(" Switch OPEN (OFF)")
        time.sleep(0.2)

finally:
    GPIO.cleanup()