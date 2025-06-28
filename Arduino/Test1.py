import serial
import time

class MotorTester:
    def __init__(self, port='/dev/ttyACM0', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None

    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
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

    def test_motor(self):
        if not self.connect():
            print("Failed to connect to Arduino")
            return

        print("\nMotor Test Program")
        print("==================")

        while True:
            print("\nCommands:")
            print("1 - Forward")
            print("2 - Backward")
            print("3 - Stop")
            print("4 - Run Test Sequence")
            print("5 - Exit")

            choice = input("\nEnter command (1-5): ")

            if choice == '1':
                self.send_command('F')
            elif choice == '2':
                self.send_command('B')
            elif choice == '3':
                self.send_command('S')
            elif choice == '4':
                print("Running test sequence...")
                self.send_command('T')
                # Wait for test sequence to complete
                time.sleep(6)
            elif choice == '5':
                print("Stopping motor and exiting...")
                self.send_command('S')
                self.close()
                break
            else:
                print("Invalid command!")

def main():
    tester = MotorTester()
    try:
        tester.test_motor()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        tester.send_command('S')  # Stop motor
        tester.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            tester.send_command('S')  # Try to stop motor
            tester.close()
        except:
            pass

if __name__ == "__main__":
    main()