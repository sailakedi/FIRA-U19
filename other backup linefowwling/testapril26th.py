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
        self.TURN_DURATION = 0.8
        self.prev_avg_centers = []
        self.max_centers = 5
        self.last_valid_command = 'S'
        self.line_lost_time = None
        self.LINE_LOST_TIMEOUT = 2.0
        self.is_u_turning = False
        self.u_turn_start_time = None
        self.U_TURN_DURATION = 1.0

        self.pre_u_turn_delay = False
        self.pre_u_turn_start_time = None
        self.PRE_U_TURN_DURATION = 1.0  # 1s forward before U-turn
        
        # Red line detection variables
        self.red_detected = False
        self.red_detected_time = None
        self.RED_DETECTION_DELAY = 1.1
        self.stopped_for_red = False
       
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
    left_marker = False
    right_marker = False
    left_marker_above = False
    right_marker_above = False
    frame_center = frame_width // 2
   
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            green_center_x = x + w//2
            green_center_y = y + h//2
           
            if green_center_x < frame_center:
                if green_center_y > black_line_y:
                    left_marker = True
                else:
                    left_marker_above = True
            else:
                if green_center_y > black_line_y:
                    right_marker = True
                else:
                    right_marker_above = True
   
    if left_marker and right_marker:
        return True, False
    elif left_marker_above and right_marker_above:
        return False, True
    elif left_marker_above or right_marker_above:
        return False, True
    return False, False

def process_frame(frame, line_tracker):
    height = frame.shape[0]
    roi_top = int(height * 0.4)
    roi_bottom = int(height * 0.7)
    roi = frame[roi_top:roi_bottom, :]
   
    blur = cv2.blur(roi, (5,5))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
   
    black_mask = cv2.inRange(hsv,
                            np.array([0,0,0]),
                            np.array([180,255,50]))
   
    low_green = np.array([35,80,60])
    up_green = np.array([85,255,255])
    green_mask = cv2.inRange(hsv, low_green, up_green)

    low_red1 = np.array([0,100,100])
    up_red1 = np.array([10,255,255])
    low_red2 = np.array([160,100,100])
    up_red2 = np.array([180,255,255])
    red_mask1 = cv2.inRange(hsv, low_red1, up_red1)
    red_mask2 = cv2.inRange(hsv, low_red2, up_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
   
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
   
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    command = 'S'
    center_points = []
    center_point = None
    largest_contour = None
    green_detected = False
    green_position = None
    line_detected = False
    current_time = time.time()
    should_continue = False
    red_status = ""

    red_detected = False
    for contour in red_contours:
        if cv2.contourArea(contour) > 500:
            red_detected = True
            if not line_tracker.red_detected:
                line_tracker.red_detected = True
                line_tracker.red_detected_time = current_time
                print("Red line detected! Will stop in 2 seconds.")
            break

    if red_detected or line_tracker.red_detected:
        time_since_red = current_time - line_tracker.red_detected_time if line_tracker.red_detected_time else 0
        if time_since_red < line_tracker.RED_DETECTION_DELAY:
            red_status = f"RED DETECTED - Stopping in {line_tracker.RED_DETECTION_DELAY - time_since_red:.1f}s"
        elif not line_tracker.stopped_for_red:
            command = 'S'
            line_tracker.stopped_for_red = True
            red_status = "STOPPED DUE TO RED LINE"
            display_roi = roi.copy()
            for contour in red_contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return display_roi, black_mask, command, None, None, False, None, red_status, red_mask

    if black_contours:
        largest_contour = max(black_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 100:
            line_detected = True
            x, y, w, h = cv2.boundingRect(largest_contour)
            heights = [y + h//4, y + h//2, y + 3*h//4]
            for height in heights:
                points_at_height = [point[0] for point in largest_contour[:, 0, :] if abs(point[1] - height) < 2]
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

                if not line_tracker.is_turning and not line_tracker.is_u_turning and not line_tracker.pre_u_turn_delay:
                    should_u_turn, should_continue = check_green_markers(green_contours, smooth_y, roi.shape[1])
                    if should_u_turn:
                        line_tracker.pre_u_turn_delay = True
                        line_tracker.pre_u_turn_start_time = current_time
                        command = 'F'
                        print("Pre U-turn delay started: Moving forward for 1s")
                    elif should_continue:
                        command = 'F'
                        print("Green markers above - continuing straight")
                    else:
                        for contour in green_contours:
                            if cv2.contourArea(contour) > 500:
                                x, y, w, h = cv2.boundingRect(contour)
                                green_center_x = x + w//2
                                green_center_y = y + h//2
                                if green_center_y > smooth_y:
                                    green_detected = True
                                    green_position = 'left' if green_center_x < smooth_x else 'right'
                                    line_tracker.is_turning = True
                                    line_tracker.turn_start_time = current_time
                                    line_tracker.turn_direction = green_position
                                    break

                if line_tracker.pre_u_turn_delay:
                    command = 'F'
                    if current_time - line_tracker.pre_u_turn_start_time >= line_tracker.PRE_U_TURN_DURATION:
                        line_tracker.pre_u_turn_delay = False
                        line_tracker.is_u_turning = True
                        line_tracker.u_turn_start_time = current_time
                        command = 'U'
                        print("Now performing U-turn")

                elif line_tracker.is_u_turning:
                    command = 'U'
                    if current_time - line_tracker.u_turn_start_time >= line_tracker.U_TURN_DURATION:
                        line_tracker.is_u_turning = False

                elif line_tracker.is_turning:
                    command = 'L' if line_tracker.turn_direction == 'left' else 'R'
                    if current_time - line_tracker.turn_start_time >= line_tracker.TURN_DURATION:
                        line_tracker.is_turning = False

                else:
                    diff = smooth_x - roi.shape[1] // 2
                    if abs(diff) < 40:
                        command = 'F'
                    elif diff < 0:
                        command = 'L'
                    else:
                        command = 'R'
            line_tracker.last_valid_command = command
            line_tracker.line_lost_time = None

    if not line_detected and not line_tracker.stopped_for_red:
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
    if center_point:
        cv2.circle(display_roi, center_point, 5, (0,0,255), -1)
    if largest_contour is not None:
        cv2.drawContours(display_roi, [largest_contour], 0, (255,0,0), 2)
    for contour in green_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for contour in red_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

    turn_status = ""
    if red_status:
        turn_status = red_status
    elif line_tracker.is_u_turning:
        turn_status = "U-Turn"
    elif line_tracker.is_turning:
        turn_status = f"Turning {line_tracker.turn_direction}"
    elif should_continue:
        turn_status = "Continue Straight"
    elif line_tracker.pre_u_turn_delay:
        turn_status = "Forward before U-turn"

    if not line_detected and not line_tracker.stopped_for_red:
        cv2.putText(display_roi, "Line Lost!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return display_roi, black_mask, command, center_point, largest_contour, green_detected, green_position, turn_status, red_mask

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

                display_roi, black_mask, command, center_point, largest_contour, green_detected, green_position, turn_status, red_mask = process_frame(frame, line_tracker)
                line_tracker.send_command(command)

                cv2.putText(display_roi, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
                cv2.putText(display_roi, f"CMD: {command}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if turn_status:
                    cv2.putText(display_roi, turn_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Line Detection", display_roi)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('r'):
                    line_tracker.red_detected = False
                    line_tracker.stopped_for_red = False
                    print("Red detection reset")
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
