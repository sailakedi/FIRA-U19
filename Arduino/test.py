import serial
import time

class RobotController:
    def __init__(self, port='/dev/ttyACM0', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None

    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print("Successfully connected to Arduino")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            return False

    def send_command(self, command):
        if self.arduino and self.arduino.is_open:
            self.arduino.write(command.encode())
            # Read and print Arduino response
            response = self.arduino.readline().decode().strip()
            print(f"Arduino response: {response}")
            return True
        return False

    def close(self):
        if self.arduino:
            self.arduino.close()
            print("Connection closed")

    def control_robot(self):
        if not self.connect():
            print("Failed to connect to Arduino")
            return

        print("\nRobot Control Program")
        print("=====================")

        while True:
            print("\nCommands:")
            print("1 - Forward")
            print("2 - Backward")
            print("3 - Turn Left")
            print("4 - Turn Right")
            print("5 - Stop")
            print("6 - Run Test Sequence")
            print("7 - Exit")
            print("\nAlternatively, use:")
            print("F - Forward")
            print("B - Backward")
            print("L - Turn Left")
            print("R - Turn Right")
            print("S - Stop")
            print("T - Test Sequence")

            choice = input("\nEnter command: ").upper()

            if choice in ['1', 'F']:
                self.send_command('F')
                print("Moving Forward")
            
            elif choice in ['2', 'B']:
                self.send_command('B')
                print("Moving Backward")
            
            elif choice in ['3', 'L']:
                self.send_command('L')
                print("Turning Left")
            
            elif choice in ['4', 'R']:
                self.send_command('R')
                print("Turning Right")
            
            elif choice in ['5', 'S']:
                self.send_command('S')
                print("Stopping")
            
            elif choice in ['6', 'T']:
                print("Running test sequence...")
                self.send_command('T')
                # Wait for test sequence to complete
                # Forward (2s) + Stop (1s) + Backward (2s) + Stop (1s) + 
                # Left (1s) + Stop (1s) + Right (1s) + Stop
                time.sleep(10)
                print("Test sequence completed")
            
            elif choice in ['7', 'EXIT', 'Q', 'QUIT']:
                print("Stopping motors and exiting...")
                self.send_command('S')
                self.close()
                break
            
            else:
                print("Invalid command!")

def main():
    controller = RobotController()
    try:
        controller.control_robot()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        controller.send_command('S')  # Stop all motors
        controller.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            controller.send_command('S')  # Try to stop all motors
            controller.close()
        except:
            pass

if __name__ == "__main__":
    main()