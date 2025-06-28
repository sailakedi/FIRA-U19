import sys
import numpy as np
import cv2
import serial
import time
from datetime import datetime
import gc

debug_mode = True

class LineTracker:
    def __init__(self):
        self.previous_center = None
        self.smooth_factor = 0.8
        self.last_command = None
        self.frame_count = 0
        self.is_turning = False
        self.turn_start_time = None
        self.turn_direction = None
        self.TURN_DURATION = 1.0
        self.MAX_TURN_DURATION = 2.0
        self.prev_avg_centers = []
        self.max_centers = 5
        self.last_valid_command = 'S'
        self.line_lost_time = None
        self.LINE_LOST_TIMEOUT = 2.0
        self.turn_history = []
        self.turn_history_max = 3
        
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
                
    def check_turn_timeout(self, current_time):
        if self.is_turning and current_time - self.turn_start_time > self.MAX_TURN_DURATION:
            self.is_turning = False
            self.turn_start_time = None
            self.turn_direction = None
            return True
        return False
        
    def add_turn_detection(self, turn_left, turn_right):
        self.turn_history.append((turn_left, turn_right))
        if len(self.turn_history) > self.turn_history_max:
            self.turn_history.pop(0)
            
        if len(self.turn_history) == self.turn_history_max:
            left_count = sum(t[0] for t in self.turn_history)
            right_count = sum(t[1] for t in self.turn_history)
            if left_count == self.turn_history_max:
                return 'left'
            elif right_count == self.turn_history_max:
                return 'right'
        return None

def get_black_mask(hsv_image):
    height = hsv_image.shape[0]
    black_mask = cv2.inRange(hsv_image, np.array([0,0,0]), np.array([180,255,135]))
    black_mask[0:int(height * 0.4), :] = cv2.inRange(
        hsv_image[0:int(height * 0.4), :],
        np.array([0,0,0]),
        np.array([180,255,90])
    )
    return black_mask

def check_green(contours_grn, black_mask, display_roi):
    black_around_sign = np.zeros((len(contours_grn), 5), dtype=np.int16)
    
    for i, contour in enumerate(contours_grn):
        area = cv2.contourArea(contour)
        if area <= 2500:
            continue
            
        green_box = cv2.boxPoints(cv2.minAreaRect(contour))
        draw_box = np.intp(green_box)
        
        w = np.linalg.norm(green_box[0] - green_box[1])
        h = np.linalg.norm(green_box[1] - green_box[2])
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.5:
            continue
            
        cv2.drawContours(display_roi, [draw_box], -1, (0, 0, 255), 2)
        
        green_box = green_box[green_box[:, 1].argsort()]
        marker_height = green_box[-1][1] - green_box[0][1]
        
        top_y = int(green_box[1][1])
        top_x_min = int(min(green_box[0:2, 0]))
        top_x_max = int(max(green_box[0:2, 0]))
        roi_t = black_mask[max(0, top_y-int(marker_height*0.8)):top_y,
                         top_x_min:top_x_max]
        if roi_t.size > 0 and np.mean(roi_t) > 125:
            black_around_sign[i, 1] = 1
        
        left_x = int(min(green_box[:, 0]))
        roi_l = black_mask[top_y:int(top_y+marker_height),
                         max(0, left_x-int(marker_height*0.8)):left_x]
        if roi_l.size > 0 and np.mean(roi_l) > 125:
            black_around_sign[i, 2] = 1
        
        right_x = int(max(green_box[:, 0]))
        roi_r = black_mask[top_y:int(top_y+marker_height),
                         right_x:min(right_x+int(marker_height*0.8), black_mask.shape[1])]
        if roi_r.size > 0 and np.mean(roi_r) > 125:
            black_around_sign[i, 3] = 1
            
        black_around_sign[i, 4] = int(green_box[2][1])
        
    return determine_turn_direction(black_around_sign, black_mask.shape[0])

def determine_turn_direction(black_around_sign, roi_height):
    turn_left = False
    turn_right = False
    left_bottom = False
    right_bottom = False
    
    valid_signs = 0
    for i in black_around_sign:
        if np.sum(i[:4]) == 2:
            valid_signs += 1
            if i[1] == 1 and i[2] == 1:  # Top and left
                turn_right = True
                if i[4] > roi_height * 0.95:
                    right_bottom = True
            elif i[1] == 1 and i[3] == 1:  # Top and right
                turn_left = True
                if i[4] > roi_height * 0.95:
                    left_bottom = True
    
    if valid_signs == 0:
        return False, False, False, False
                    
    return turn_left, turn_right, left_bottom, right_bottom

def process_frame(frame, line_tracker):
    height = frame.shape[0]
    roi_height = int(height * 0.7)
    roi = frame[height-roi_height:height, :]
    
    blur = cv2.blur(roi, (5,5))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    black_mask = get_black_mask(hsv)
    
    green_min = np.array([40, 50, 45])
    green_max = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_min, green_max)
    
    kernel = np.ones((3,3), np.uint8)
    
    black_mask = cv2.erode(black_mask, kernel, iterations=5)
    black_mask = cv2.dilate(black_mask, kernel, iterations=17)
    black_mask = cv2.erode(black_mask, kernel, iterations=9)
    
    green_mask = cv2.erode(green_mask, kernel, iterations=1)
    green_mask = cv2.dilate(green_mask, kernel, iterations=11)
    green_mask = cv2.erode(green_mask, kernel, iterations=9)
    
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
    line_detected = False
    current_time = time.time()
    
    if black_contours:
        largest_contour = max(black_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        min_line_size = 100
        if area > min_line_size:
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
                
                display_roi = roi.copy()
                
                if not line_tracker.is_turning and len(green_contours) > 0:
                    turn_left, turn_right, left_bottom, right_bottom = check_green(green_contours, black_mask, display_roi)
                    
                    turn_direction = line_tracker.add_turn_detection(turn_left, turn_right)
                    if turn_direction:
                        line_tracker.is_turning = True
                        line_tracker.turn_start_time = current_time
                        line_tracker.turn_direction = turn_direction
                
                line_tracker.check_turn_timeout(current_time)
                
                if line_tracker.is_turning:
                    command = 'L' if line_tracker.turn_direction == 'left' else 'R'
                    
                    if current_time - line_tracker.turn_start_time >= line_tracker.TURN_DURATION:
                        line_tracker.is_turning = False
                        line_tracker.turn_start_time = None
                        line_tracker.turn_direction = None
                else:
                    center_threshold = 40
                    frame_center = roi.shape[1] // 2
                    diff = smooth_x - frame_center
                    
                    if abs(diff) < center_threshold:
                        command = 'F'
                    elif diff < 0:
                        command = 'L'
                    else:
                        command = 'R'
                        
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
        if area > 2500:
            cv2.drawContours(display_roi, [np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))], 0, (0, 255, 0), 2)
    
    turn_status = ""
    if line_tracker.is_turning:
        turn_status = f"Turning {line_tracker.turn_direction}"
        
    if not line_detected:
        cv2.putText(display_roi, "Line Lost!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if debug_mode:
        cv2.putText(display_roi, f"Black Area: {area:.0f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_roi, f"Green Signs: {len(green_contours)}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if center_point:
            cv2.putText(display_roi, f"Pos: ({center_point[0]}, {center_point[1]})", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display_roi, black_mask, command, center_point, largest_contour, turn_status

def show_camera():
    camera_id = "/dev/video1"
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
	
