import Jetson.GPIO as GPIO
import time

# Set the GPIO mode to BOARD
GPIO.setmode(GPIO.BOARD)

# Define the switch pin (pin 7)
SWITCH_PIN = 15

# Set up the GPIO pin as an input with internal pull-up resistor
# This line activates the internal pull-up resistor, so you don't need a physical one
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#try:
    # print("Switch state monitoring started. Press CTRL+C to exit.")
    # print("Direct switch connection between Pin 7 and GND")
    
    # while True:
    #     # Read the switch state
while True:
    
    switch_state = GPIO.input(SWITCH_PIN)
    print(switch_state)

    #     # Display the state
    #     if switch_state:
    #         print("Switch: ON ", end="\r")
    #     else:
    #         print("Switch: OFF", end="\r")
            
    #     time.sleep(0.1)



# except KeyboardInterrupt:
#     print("\nProgram stopped by user")

# finally:
#     # Clean up GPIO settings
#     GPIO.cleanup()