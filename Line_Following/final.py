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
        self.TURN_DURATION = 0.6
        self.prev_avg_centers = []
        self.max_centers = 5
        self.last_valid_command = 'S'
        self.line_lost_time = None
        self.LINE_LOST_TIMEOUT = 2.0
        self.is_u_turning = False
        self.u_turn_start_time = None
        self.U_TURN_DURATION = 1.3
        self.u_turn_delay_start = None
        self.U_TURN_DELAY = 0.5  # Reduced from 1.0 second for quicker response
        
        try:
            self.arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            time.sleep(2)
            self.send_command('S')
            print("Arduino connected successfully")
        except:
            print("Failed to connect to Arduino")
            self.arduino = None

    def send_command(self, command):
        if command != self.last_command:
            if self.arduino:
                self.arduino.write(command.encode())
                print(f"Sent command: {command}")
            self.last_command = command

def check_green_markers(green_contours, black_line_y, frame_width):
    left_markers = []
    right_markers = []
    left_markers_above = []
    right_markers_above = []
    frame_center = frame_width // 2
    
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Reduced threshold for better sensitivity
            x, y, w, h = cv2.boundingRect(contour)
            green_center_x = x + w//2
            green_center_y = y + h//2
            
            # Determine if marker is left or right of center
            if green_center_x < frame_center:  # Left side
                if green_center_y > black_line_y:  # Below black line
                    left_markers.append((x, y, w, h))
                else:  # Above black line
                    left_markers_above.append((x, y, w, h))
            else:  # Right side
                if green_center_y > black_line_y:  # Below black line
                    right_markers.append((x, y, w, h))
                else:  # Above black line
                    right_markers_above.append((x, y, w, h))
    
    # If we have at least one marker on each side below the line, it's a U-turn
    if left_markers and right_markers:
        print(f"U-TURN: {len(left_markers)} left markers, {len(right_markers)} right markers")
        return True, False
    elif left_markers_above and right_markers_above:  # Both markers above - continue straight
        return False, True
    elif left_markers_above or right_markers_above:  # Any marker above - continue straight
        return False, True
    
    return False, False

def process_frame(frame, line_tracker):
    height = frame.shape[0]
    # Adjust ROI to focus more on the area below
    roi_top = int(height * 0.5)  # Changed from 0.4 to 0.5 to focus more on the bottom
    roi_bottom = int(height * 0.7)  # Changed from 0.7 to 0.8 to see more of the ground
    roi = frame[roi_top:roi_bottom, :]
    
    blur = cv2.blur(roi, (5,5))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    black_mask = cv2.inRange(hsv, 
                            np.array([0,0,0]), 
                            np.array([180,255,50]))
    
    # Improved green detection range
    low_green = np.array([35,50,50])  # Lower saturation and value thresholds
    up_green = np.array([90,255,255]) # Slightly wider hue range
    green_mask = cv2.inRange(hsv, low_green, up_green)
    
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    black_contours, _ = cv2.findContours(black_mask, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)
    
    green_contours, _ = cv2.findContours(green_mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    command = 'S'
    center_points = []
    center_point = None
    largest_contour = None
    green_detected = False
    green_position = None
    line_detected = False
    current_time = time.time()
    should_continue = False
    
    if black_contours:
        largest_contour = max(black_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 100:
            line_detected = True
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            heights = [y + h//4, y + h//2, y + 3*h//4]
            
            for height in heights:
                points_at_height = []
                for point in largest_contour[:, 0, :]:
                    if abs(point[1] - height) < 2:
                        points_at_height.append(point[0])
                
                if points_at_height:
                    center_x = (min(points_at_height) + max(points_at_height)) // 2
                    center_points.append((center_x, height))
            
            if center_points:
                avg_x = int(sum(p[0] for p in center_points) / len(center_points))
                avg_y = int(sum(p[1] for p in center_points) / len(center_points))
                
                line_tracker.prev_avg_centers.append((avg_x, avg_y))
                if len(line_tracker.prev_avg_centers) > line_tracker.max_centers:
                    line_tracker.prev_avg_centers.pop(0)
                
                smooth_x = int(sum(p[0] for p in line_tracker.prev_avg_centers) / len(line_tracker.prev_avg_centers))
                smooth_y = int(sum(p[1] for p in line_tracker.prev_avg_centers) / len(line_tracker.prev_avg_centers))
                center_point = (smooth_x, smooth_y)
                
                # Improved green marker detection and U-turn logic
                if not line_tracker.is_turning and not line_tracker.is_u_turning and line_tracker.u_turn_delay_start is None:
                    should_u_turn, should_continue = check_green_markers(green_contours, smooth_y, roi.shape[1])
                    
                    if should_u_turn:
                        # Start the delay before U-turn
                        line_tracker.u_turn_delay_start = current_time
                        command = 'S'  # Stop during the delay
                        print("U-turn detected! Waiting for delay...")
                    elif should_continue:
                        command = 'F'
                        print("Green markers above - continuing straight")
                    else:
                        # Only check for individual green markers if we're not doing a U-turn
                        for contour in green_contours:
                            area = cv2.contourArea(contour)
                            if area > 300:  # Reduced from 500 for better sensitivity
                                x, y, w, h = cv2.boundingRect(contour)
                                green_center_x = x + w//2
                                green_center_y = y + h//2
                                
                                if green_center_y > smooth_y:
                                    green_detected = True
                                    if green_center_x < smooth_x:
                                        green_position = 'left'
                                    else:
                                        green_position = 'right'
                                    
                                    line_tracker.is_turning = True
                                    line_tracker.turn_start_time = current_time
                                    line_tracker.turn_direction = green_position
                                    break
                
                # Prioritized command execution
                # Check if we're in the U-turn delay period
                if line_tracker.u_turn_delay_start is not None:
                    if current_time - line_tracker.u_turn_delay_start >= line_tracker.U_TURN_DELAY:
                        # Delay is over, start the U-turn
                        line_tracker.is_u_turning = True
                        line_tracker.u_turn_start_time = current_time
                        line_tracker.u_turn_delay_start = None
                        command = 'U'
                        print("U-turn delay complete, starting U-turn!")
                    else:
                        # Still in delay period
                        command = 'S'  # Stop during the delay
                elif line_tracker.is_u_turning:
                    command = 'U'
                    if current_time - line_tracker.u_turn_start_time >= line_tracker.U_TURN_DURATION:
                        line_tracker.is_u_turning = False
                        line_tracker.u_turn_start_time = None
                elif line_tracker.is_turning:
                    command = 'L' if line_tracker.turn_direction == 'left' else 'R'
                    if current_time - line_tracker.turn_start_time >= line_tracker.TURN_DURATION:
                        line_tracker.is_turning = False
                        line_tracker.turn_start_time = None
                        line_tracker.turn_direction = None
                else:
                    if center_point is not None:
                        center_threshold = 40
                        frame_center = roi.shape[1] // 2
                        diff = smooth_x - frame_center
                        
                        if abs(diff) < center_threshold:
                            command = 'F'
                        elif diff < 0:
                            command = 'L'
                        else:
                            command = 'R'
    
    if line_detected:
        line_tracker.last_valid_command = command
        line_tracker.line_lost_time = None
    
    if not line_detected:
        if line_tracker.line_lost_time is None:
            line_tracker.line_lost_time = current_time
        
        if current_time - line_tracker.line_lost_time < line_tracker.LINE_LOST_TIMEOUT:
            command = line_tracker.last_valid_command
            print("Line lost, continuing last command:", command)
        else:
            command = 'S'
            print("Line lost timeout, stopping")
    
    display_roi = roi.copy()
    
    for point in center_points:
        cv2.circle(display_roi, point, 3, (0,255,255), -1)
    
    if center_point is not None:
        cv2.circle(display_roi, center_point, 5, (0,0,255), -1)
    
    if largest_contour is not None:
        cv2.drawContours(display_roi, [largest_contour], 0, (255,0,0), 2)
    
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Reduced from 500 for better sensitivity
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if center_point is not None and y > center_point[1]:
                cv2.putText(display_roi, 'Green Below', (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display_roi, 'Green Above', (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    turn_status = ""
    if line_tracker.u_turn_delay_start is not None:
        delay_remaining = line_tracker.U_TURN_DELAY - (current_time - line_tracker.u_turn_delay_start)
        turn_status = f"U-Turn Delay: {delay_remaining:.1f}s"
    elif line_tracker.is_u_turning:
        turn_status = "U-Turn"
    elif line_tracker.is_turning:
        turn_status = f"Turning {line_tracker.turn_direction}"
    elif should_continue:
        turn_status = "Continue Straight"
    
    if not line_detected:
        cv2.putText(display_roi, "Line Lost!", (10, 120),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add debug info for green markers
    cv2.putText(display_roi, f"Green markers: {len(green_contours)}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return display_roi, black_mask, command, center_point, largest_contour, green_detected, green_position, turn_status

def show_camera():
    camera_id = "/dev/video0"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    # Make the camera view narrower
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Changed from 320 to 280
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
                
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_time)
                    last_time = current_time
                    print(f"FPS: {fps:.1f}")
                    gc.collect()
                
                display_roi, black_mask, command, center_point, largest_contour, green_detected, green_position, turn_status = process_frame(frame, line_tracker)
                
                line_tracker.send_command(command)
                
                cv2.putText(display_roi, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                cv2.putText(display_roi, f"CMD: {command}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if turn_status:
                    cv2.putText(display_roi, turn_status, (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Line Detection", display_roi)
                #cv2.imshow("Black Mask", black_mask)
                
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