import sys
import numpy as np
import cv2
import serial
import time
from datetime import datetime
import gc

class LineTracker:
    def __init__(self):
        self.previous_center = None
        self.smooth_factor = 0.8
        self.last_command = None
        self.frame_count = 0
        self.is_turning = False
        self.turn_start_time = None
        self.turn_direction = None
        self.TURN_DURATION = 1.0  # Duration for 90-degree turn
        
        try:
            self.arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            time.sleep(2)
            self.send_command('S')
            print("Arduino connected successfully")
        except:
            print("Failed to connect to Arduino")
            self.arduino = None

    def send_command(self, command):
        if command != self.last_command:  # Only send if command changes
            if self.arduino:
                self.arduino.write(command.encode())
                print(f"Sent command: {command}")
                self.last_command = command

def process_frame(frame, line_tracker):
    height = frame.shape[0]
    roi_height = int(height * 0.4)
    roi = frame[height-roi_height:height, :]
    
    blur = cv2.blur(roi, (5,5))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Black line detection
    black_mask = cv2.inRange(hsv, 
                            np.array([0,0,0]), 
                            np.array([180,255,50]))
    
    # Green detection
    low_green = np.array([50,100,50])
    up_green = np.array([70,255,255])
    green_mask = cv2.inRange(hsv, low_green, up_green)
    
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    black_contours, _ = cv2.findContours(black_mask, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    
    green_contours, _ = cv2.findContours(green_mask,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    
    command = 'S'
    center_point = None
    largest_contour = None
    green_detected = False
    green_position = None
    current_time = time.time()
    
    if black_contours:
        largest_contour = max(black_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 100:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M['m10']/M["m00"])
                cy = int(M['m01']/M["m00"])
                center_point = (cx, cy)
                
                # Check for green objects if not already turning
                if not line_tracker.is_turning:
                    for contour in green_contours:
                        area = cv2.contourArea(contour)
                        if area > 500:
                            x, y, w, h = cv2.boundingRect(contour)
                            green_center_x = x + w//2
                            
                            if y > cy:  # Green is below the line
                                green_detected = True
                                if green_center_x < cx:
                                    green_position = 'left'
                                else:
                                    green_position = 'right'
                                
                                # Start turning immediately
                                line_tracker.is_turning = True
                                line_tracker.turn_start_time = current_time
                                line_tracker.turn_direction = green_position
                
                # Decision logic
                if line_tracker.is_turning:
                    # Execute 90-degree turn
                    command = 'L' if line_tracker.turn_direction == 'left' else 'R'
                    
                    # Check if turn is complete
                    if current_time - line_tracker.turn_start_time >= line_tracker.TURN_DURATION:
                        line_tracker.is_turning = False
                        line_tracker.turn_start_time = None
                        line_tracker.turn_direction = None
                else:
                    # Normal line following
                    center_threshold = 40
                    frame_center = roi.shape[1] // 2
                    diff = cx - frame_center
                    
                    if abs(diff) < center_threshold:
                        command = 'F'
                    elif diff < 0:
                        command = 'L'
                    else:
                        command = 'R'
    
    # Prepare visualization
    display_roi = roi.copy()
    
    # Draw black line detection
    if center_point is not None:
        cv2.circle(display_roi, center_point, 5, (0,0,255), -1)
    if largest_contour is not None:
        cv2.drawContours(display_roi, [largest_contour], 0, (255,0,0), 2)
    
    # Draw green objects
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_roi, 'Green Object', (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add turning status
    turn_status = ""
    if line_tracker.is_turning:
        turn_status = f"Turning {line_tracker.turn_direction}"
    
    return display_roi, black_mask, command, center_point, largest_contour, green_detected, green_position, turn_status

def show_camera():
    camera_id = "/dev/video0"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    # Set camera parameters
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    line_tracker = LineTracker()
    frame_count = 0
    last_time = time.time()
    fps = 0
    
    if video_capture.isOpened():
        try:
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    continue

                frame_count += 1
                
                # Calculate FPS every 30 frames
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_time)
                    last_time = current_time
                    print(f"FPS: {fps:.1f}")
                    gc.collect()
                
                # Process frame
                display_roi, black_mask, command, center_point, largest_contour, green_detected, green_position, turn_status = process_frame(frame, line_tracker)
                
                # Send command to Arduino
                line_tracker.send_command(command)
                
                # Add FPS and command text
                cv2.putText(display_roi, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_roi, f"CMD: {command}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if turn_status:
                    cv2.putText(display_roi, turn_status, (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show images
                cv2.imshow("Line Detection", display_roi)
                cv2.imshow("Black Mask", black_mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        finally:
            line_tracker.send_command('S')
            video_capture.release()
            cv2.destroyAllWindows()
            if line_tracker.arduino:
                line_tracker.arduino.close()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    show_camera()