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
        self.U_TURN_DELAY = 0.5
        # Obstacle avoidance parameters
        self.avoiding_obstacle = False
        self.obstacle_avoidance_start = None
        self.obstacle_phase = 0
        self.TURN_90_DURATION = 1.0  # Adjust for 90-degree turn
        self.FORWARD_1S_DURATION = 0.5
        self.FORWARD_2S_DURATION = 1

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

def detect_obstacle(frame, min_area=500):
    """
    Detect obstacles using edge detection and contour analysis
    Returns: (bool, bbox) - detection status and bounding box if detected
    """
    height = frame.shape[0]
    roi_top = int(height * 0.3)
    roi_bottom = int(height * 0.5)
    roi = frame[roi_top:roi_bottom, :]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_center = roi.shape[1] // 2
    center_margin = roi.shape[1] // 4
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            contour_center = x + w//2
            
            if (contour_center > frame_center - center_margin and 
                contour_center < frame_center + center_margin):
                
                aspect_ratio = float(w)/h if h != 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    return True, (x, y, w, h)
    
    return False, None

def check_green_markers(green_contours, black_mask, frame_width, frame_height):
    """
    Check green markers and their relation to black areas around them
    Returns: (should_u_turn, turn_direction)
    """
    if not green_contours:
        return False, None
    
    turn_left = False
    turn_right = False
    left_bottom = False
    right_bottom = False
    
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area <= 300:
            continue
            
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        box = box[box[:, 1].argsort()]
        marker_height = box[-1][1] - box[0][1]
        
        is_at_bottom = box[2][1] > frame_height * 0.95
        
        box_x_sorted = box[box[:, 0].argsort()]
        
        top_y1 = max(int(box[0][1] - marker_height * 0.8), 0)
        top_y2 = int(box[0][1])
        top_x1 = min(int(box[0][0]), int(box[1][0]))
        top_x2 = max(int(box[0][0]), int(box[1][0]))
        
        left_x1 = max(int(box_x_sorted[0][0] - marker_height * 0.8), 0)
        left_x2 = int(box_x_sorted[0][0])
        left_y1 = min(int(box_x_sorted[0][1]), int(box_x_sorted[1][1]))
        left_y2 = max(int(box_x_sorted[0][1]), int(box_x_sorted[1][1]))
        
        right_x1 = int(box_x_sorted[2][0])
        right_x2 = min(int(box_x_sorted[2][0] + marker_height * 0.8), frame_width)
        right_y1 = min(int(box_x_sorted[2][1]), int(box_x_sorted[3][1]))
        right_y2 = max(int(box_x_sorted[2][1]), int(box_x_sorted[3][1]))
        
        black_top = False
        black_left = False
        black_right = False
        
        if top_y1 < top_y2 and top_x1 < top_x2:
            top_region = black_mask[top_y1:top_y2, top_x1:top_x2]
            if top_region.size > 0 and np.mean(top_region) > 125:
                black_top = True
                
        if left_y1 < left_y2 and left_x1 < left_x2:
            left_region = black_mask[left_y1:left_y2, left_x1:left_x2]
            if left_region.size > 0 and np.mean(left_region) > 125:
                black_left = True
                
        if right_y1 < right_y2 and right_x1 < right_x2:
            right_region = black_mask[right_y1:right_y2, right_x1:right_x2]
            if right_region.size > 0 and np.mean(right_region) > 125:
                black_right = True
        
        if black_top and black_left and not black_right:
            turn_right = True
            if is_at_bottom:
                right_bottom = True
        elif black_top and black_right and not black_left:
            turn_left = True
            if is_at_bottom:
                left_bottom = True
    
    if turn_left and turn_right and not (left_bottom and right_bottom):
        return True, None
    elif turn_left and not turn_right and not left_bottom:
        return False, 'left'
    elif turn_right and not turn_left and not right_bottom:
        return False, 'right'
    else:
        return False, None

def process_frame(frame, line_tracker):
    height = frame.shape[0]
    roi_top = int(height * 0.5)
    roi_bottom = int(height * 0.7)
    roi = frame[roi_top:roi_bottom, :]
    
    blur = cv2.blur(roi, (5,5))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    black_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,50]))
    green_mask = cv2.inRange(hsv, np.array([35,50,50]), np.array([90,255,255]))
    
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    command = 'S'
    center_points = []
    center_point = None
    largest_contour = None
    line_detected = False
    current_time = time.time()
    turn_status = ""
    
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

    # Obstacle detection and avoidance
    obstacle_detected, obstacle_bbox = detect_obstacle(frame)
    
    if obstacle_detected and not line_tracker.avoiding_obstacle and not line_tracker.is_turning and not line_tracker.is_u_turning:
        line_tracker.avoiding_obstacle = True
        line_tracker.obstacle_avoidance_start = current_time
        line_tracker.obstacle_phase = 0
        command = 'S'
        turn_status = "Obstacle detected!"
    
    if line_tracker.avoiding_obstacle:
        phase_duration = current_time - line_tracker.obstacle_avoidance_start
        
        if line_tracker.obstacle_phase == 0:  # Stop
            command = 'S'
            if phase_duration >= 0.5:
                line_tracker.obstacle_phase = 1
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - stopping"
            
        elif line_tracker.obstacle_phase == 1:  # Turn right 90 degrees
            command = 'R'
            if phase_duration >= line_tracker.TURN_90_DURATION:
                line_tracker.obstacle_phase = 2
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - turning right 90"
            
        elif line_tracker.obstacle_phase == 2:  # Forward 1 second
            command = 'F'
            if phase_duration >= line_tracker.FORWARD_1S_DURATION:
                line_tracker.obstacle_phase = 3
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - forward 1"
            
        elif line_tracker.obstacle_phase == 3:  # Turn left 90 degrees
            command = 'L'
            if phase_duration >= line_tracker.TURN_90_DURATION:
                line_tracker.obstacle_phase = 4
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - turning left 90"
            
        elif line_tracker.obstacle_phase == 4:  # Forward 2 seconds
            command = 'F'
            if phase_duration >= line_tracker.FORWARD_2S_DURATION:
                line_tracker.obstacle_phase = 5
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - forward 2"
            
        elif line_tracker.obstacle_phase == 5:  # Turn left 90 degrees
            command = 'L'
            if phase_duration >= line_tracker.TURN_90_DURATION:
                line_tracker.obstacle_phase = 6
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - turning left 90"
            
        elif line_tracker.obstacle_phase == 6:  # Forward 1 second
            command = 'F'
            if phase_duration >= line_tracker.FORWARD_1S_DURATION:
                line_tracker.obstacle_phase = 0
                line_tracker.obstacle_avoidance_start = current_time
            turn_status = "Avoiding obstacle - forward 1"
            
        # elif line_tracker.obstacle_phase == 7:  # Final right turn 90 degrees
        #     command = 'R'
        #     if phase_duration >= line_tracker.TURN_90_DURATION:
        #         line_tracker.avoiding_obstacle = False
        #         line_tracker.obstacle_phase = 0
        #     turn_status = "Avoiding obstacle - final turn"

    # Green marker detection and turning logic (only if not avoiding obstacle)
    if not line_tracker.is_turning and not line_tracker.is_u_turning and line_tracker.u_turn_delay_start is None and not line_tracker.avoiding_obstacle:
        should_u_turn, turn_direction = check_green_markers(green_contours, black_mask, 
                                                          roi.shape[1], roi.shape[0])
        
        if should_u_turn:
            line_tracker.u_turn_delay_start = current_time
            command = 'S'
            turn_status = "U-turn detected!"
        elif turn_direction:
            line_tracker.is_turning = True
            line_tracker.turn_start_time = current_time
            line_tracker.turn_direction = turn_direction
            command = 'L' if turn_direction == 'left' else 'R'
            turn_status = f"Turn detected: {turn_direction}"

    # Normal line following (only if not avoiding obstacle)
    if not line_tracker.avoiding_obstacle:
        if line_tracker.u_turn_delay_start is not None:
            if current_time - line_tracker.u_turn_delay_start >= line_tracker.U_TURN_DELAY:
                line_tracker.is_u_turning = True
                line_tracker.u_turn_start_time = current_time
                line_tracker.u_turn_delay_start = None
                command = 'U'
                turn_status = "Starting U-turn!"
            else:
                command = 'S'
                turn_status = "U-turn delay"
        elif line_tracker.is_u_turning:
            command = 'U'
            if current_time - line_tracker.u_turn_start_time >= line_tracker.U_TURN_DURATION:
                line_tracker.is_u_turning = False
                line_tracker.u_turn_start_time = None
            turn_status = "U-turning"
        elif line_tracker.is_turning:
            command = 'L' if line_tracker.turn_direction == 'left' else 'R'
            if current_time - line_tracker.turn_start_time >= line_tracker.TURN_DURATION:
                line_tracker.is_turning = False
                line_tracker.turn_start_time = None
                line_tracker.turn_direction = None
            turn_status = f"Turning {line_tracker.turn_direction}"
        elif center_point is not None:
            center_threshold = 40
            frame_center = roi.shape[1] // 2
            diff = smooth_x - frame_center
            
            if abs(diff) < center_threshold:
                command = 'F'
            elif diff < 0:
                command = 'L'
            else:
                command = 'R'

    # Line lost handling
    if line_detected:
        line_tracker.last_valid_command = command
        line_tracker.line_lost_time = None
    else:
        if line_tracker.line_lost_time is None:
            line_tracker.line_lost_time = current_time
        
        if current_time - line_tracker.line_lost_time < line_tracker.LINE_LOST_TIMEOUT:
            command = line_tracker.last_valid_command
            turn_status = "Line lost, continuing last command"
        else:
            command = 'S'
            turn_status = "Line lost timeout"

    # Visualization
    display_roi = roi.copy()
    
    # Draw center points
    for point in center_points:
        cv2.circle(display_roi, point, 3, (0,255,255), -1)
    
    if center_point is not None:
        cv2.circle(display_roi, center_point, 5, (0,0,255), -1)
    
    if largest_contour is not None:
        cv2.drawContours(display_roi, [largest_contour], 0, (255,0,0), 2)
    
    # Draw green markers
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw obstacle detection visualization
    if obstacle_detected and obstacle_bbox:
        x, y, w, h = obstacle_bbox
        cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(display_roi, "Obstacle", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return display_roi, black_mask, command, center_point, largest_contour, turn_status

def show_camera():
    camera_id = "/dev/video0"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
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
                
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_time)
                    last_time = current_time
                    print(f"FPS: {fps:.1f}")
                    gc.collect()
                
                display_roi, black_mask, command, center_point, largest_contour, turn_status = process_frame(frame, line_tracker)
                
                line_tracker.send_command(command)
                
                cv2.putText(display_roi, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_roi, f"CMD: {command}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if turn_status:
                    cv2.putText(display_roi, turn_status, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Line Detection", display_roi)
                
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