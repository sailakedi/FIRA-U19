import sys
import numpy as np
import cv2
from ultralytics import YOLO
import serial
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

window_title = "USB Camera"

class ArduinoController:
    def __init__(self, port='/dev/ttyACM0', baud_rate=9600):
        try:
            self.serial = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            logger.info("Arduino connected successfully")
        except serial.SerialException as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            sys.exit(1)

    def send_command(self, command):
        try:
            self.serial.write(command.encode())
            response = self.serial.readline().decode().strip()
            logger.debug(f"Motor: {response}")
        except serial.SerialException as e:
            logger.error(f"Failed to send command: {e}")

    def close(self):
        if self.serial:
            self.serial.close()
            logger.info("Arduino connection closed")

class LineTracker:
    def __init__(self):
        self.previous_center = None
        self.smooth_factor = 0.8  # Adjust this value (0-1) for smoothing

    def update(self, contour):
        if contour is None or len(contour) == 0:
            return self.previous_center

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M['m10']/M["m00"])
            cy = int(M['m01']/M["m00"])
            current_center = (cx, cy)

            if self.previous_center is None:
                self.previous_center = current_center
            else:
                # Smooth the transition
                smooth_x = int(self.smooth_factor * self.previous_center[0] + 
                             (1 - self.smooth_factor) * current_center[0])
                smooth_y = int(self.smooth_factor * self.previous_center[1] + 
                             (1 - self.smooth_factor) * current_center[1])
                self.previous_center = (smooth_x, smooth_y)

            return self.previous_center
        return self.previous_center

def get_roi(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    roi = frame[int(height/2):height, 0:width]  # Adjust these values based on your needs
    return roi

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    
    # Add more blur to reduce noise
    blur = cv2.GaussianBlur(thresh, (9, 9), 0)
    
    # Morphological operations to fill gaps
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    
    return morph

def adaptive_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

def show_camera():
    # Initialize Arduino
    arduino = ArduinoController()
    
    # Load the YOLO model
    model = YOLO('silver_classify_s.pt')
    
    # Initialize line tracker
    line_tracker = LineTracker()
    
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video0"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
 
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            
            while True:
                ret_val, frame = video_capture.read()
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    # Create a copy of the original frame
                    display_frame = frame.copy()
                    
                    # Run YOLO inference for silver line detection
                    results = model(frame)
                    # Draw YOLO results on frame
                    display_frame = results[0].plot()

                    # Get ROI for line detection
                    roi = get_roi(frame)
                    
                    # Preprocess image
                    processed = preprocess_image(roi)
                    
                    # Process for green squares and black lines
                    blur = cv2.GaussianBlur(frame, (5,5), 0)
                    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                    
                    # Green detection
                    low_green = np.array([40,50,45])
                    up_green = np.array([85,255,255])
                    green_mask = cv2.inRange(hsv, low_green, up_green)
                    
                    # Find contours for green objects
                    green_contours, _ = cv2.findContours(green_mask, 
                                                       cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw bounding boxes around green objects
                    for contour in green_contours:
                        area = cv2.contourArea(contour)
                        if area > 500:  # Minimum area threshold to filter noise
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                                        (0, 255, 0), 2)
                            # Add label
                            cv2.putText(display_frame, 'Green Object', 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 1)
                    
                    # Black line detection with improved parameters
                    low_black = np.array([0,0,0])
                    up_black = np.array([180,255,50])
                    black_mask = cv2.inRange(hsv, low_black, up_black)
                    
                    # Apply additional morphological operations
                    kernel = np.ones((5,5), np.uint8)
                    black_mask = cv2.erode(black_mask, kernel, iterations=1)
                    black_mask = cv2.dilate(black_mask, kernel, iterations=1)

                    # Find contours for black lines
                    black_contours, _ = cv2.findContours(black_mask, 
                                                       cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Simplified motor control based on line detection
                    if len(black_contours) > 0:
                        valid_contours = [c for c in black_contours 
                                        if cv2.contourArea(c) > 100]
                        
                        if valid_contours:
                            largest_contour = max(valid_contours, 
                                                key=cv2.contourArea)
                            
                            center = line_tracker.update(largest_contour)
                            
                            if center:
                                # Draw visualization
                                cv2.circle(display_frame, center, 5, (0,0,255), -1)
                                cv2.polylines(display_frame, [largest_contour], 
                                            True, (255,0,0), 2)
                                
                                # If black line is detected, move forward
                                arduino.send_command('F')
                                cv2.putText(display_frame, "Motor: FORWARD", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 255, 0), 2)
                    else:
                        # No line detected, stop motor
                        arduino.send_command('S')
                        cv2.putText(display_frame, "Motor: STOP", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 0, 255), 2)
                    
                    # Red object detection
                    low_red1 = np.array([0, 100, 100])
                    up_red1 = np.array([10, 255, 255])
                    low_red2 = np.array([160, 100, 100])
                    up_red2 = np.array([180, 255, 255])
                    
                    red_mask1 = cv2.inRange(hsv, low_red1, up_red1)
                    red_mask2 = cv2.inRange(hsv, low_red2, up_red2)
                    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                    
                    # Apply morphological operations
                    kernel = np.ones((5,5), np.uint8)
                    red_mask = cv2.erode(red_mask, kernel, iterations=1)
                    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

                    # Find contours for red objects
                    red_contours, _ = cv2.findContours(red_mask, 
                                                     cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw bounding boxes around red objects
                    for contour in red_contours:
                        area = cv2.contourArea(contour)
                        if area > 500:  # Minimum area threshold to filter noise
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                                        (0, 0, 255), 2)  # Red color in BGR
                            # Add label
                            cv2.putText(display_frame, 'Red Object', 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 0, 255), 1)
                    
                    # Show the processed frames
                    cv2.imshow("Combined Detection", display_frame)
                    cv2.imshow("Processed ROI", processed)

                else:
                    break
                
                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    arduino.send_command('S')  # Stop motor before exiting
                    break

        finally:
            arduino.send_command('S')  # Ensure motor is stopped
            arduino.close()
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    try:
        show_camera()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)