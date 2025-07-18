import cv2
import numpy as np
import time
import os
import serial
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import torch

# Arduino communication setup
try:
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Changed baud rate to match Arduino
    print("Connected to Arduino")
    time.sleep(2)  # 给连接一些建立时间
except:
    print("Failed to connect to Arduino. Running in simulation mode.")
    arduino = None

# 相机设置
camera_width = 640
camera_height = 480
horizontal_center = camera_width // 2
text_pos = np.array([int(camera_width * 0.96), int(camera_height * 0.04)])

# 初始化球体追踪变量
ball_distance = 0
ball_type = "none"
ball_width = -1
last_best_box = None

# 机器人控制参数
distance_threshold = 50    # 认为对齐的中心像素距离阈值
proximity_threshold = 150  # 认为足够接近的球宽度像素阈值
grab_sequence_delay = 1.0  # 夹爪动作延迟时间

def send_command(command):
    """发送命令到Arduino"""
    if arduino:
        arduino.write(command.encode())
        print(f"Sent command: {command}")
        # Add delay for servo commands
        if command in ['D', 'P']:
            time.sleep(1)  # Give time for servo movement
    else:
        print(f"Simulation: {command}")

def zone_cam_loop():
    global ball_distance, ball_type, ball_width, last_best_box
    
    # 初始化FPS计数器变量
    fps_time = time.perf_counter()
    counter = 0
    fps = 0
    
    # 加载YOLO模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('ball_detect_s.pt')
    model.to(device)
    
    # 初始化相机
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    # 尝试设置更高的相机FPS
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    crop_percentage = 0.45
    crop_height = int(camera_height * crop_percentage)
    
    # 简单的移动模式
    movement_pattern = ['F', 'L', 'L', 'F']
    pattern_index = 0
    last_movement_time = time.time()
    movement_duration = 0.3  # 每个动作持续0.3秒
    
    # 设置目标为银色球
    target_ball = "silver"
    
    # 状态: "exploring"(探索), "tracking"(追踪), "grabbing"(抓取), "waiting"(等待)
    state = "exploring"
    
    # 等待计时器
    wait_start_time = 0
    wait_duration = 5  # 等待5秒
    
    # 调试信息
    detection_count = 0
    last_detection_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # 裁剪图像
        cv2_img = frame[crop_height:, :]
        
        # 运行YOLO检测
        results = model.predict(cv2_img, 
                              imgsz=(320, 160),  # 降低分辨率以提高速度
                              conf=0.25,  # 降低置信度阈值
                              iou=0.3, 
                              agnostic_nms=True, 
                              verbose=False)
        
        result = results[0].numpy()
        boxes = []
        
        # 处理检测结果
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            class_id = box.cls[0].astype(int)
            name = result.names[class_id]
            confidence = box.conf[0].astype(float)
            
            width = x2 - x1
            height = y2 - y1
            area = width * height
            distance = (x1 + width // 2) - horizontal_center
            boxes.append([area, distance, name, width, confidence])
            
            color = colors(class_id, True)
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv2_img, f"{name}: {confidence:.2f}", 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_DUPLEX, 
                       0.5, 
                       color, 
                       1, 
                       cv2.LINE_AA)
        
        # 寻找银色球
        silver_boxes = []
        for box in boxes:
            box_name = str(box[2]).lower()
            if "silver" in box_name:
                silver_boxes.append(box)
                print(f"Found silver ball: {box_name}, confidence: {box[4]:.2f}")
        
        # 状态机逻辑
        if state == "exploring":
            # 探索模式 - 按照预定模式移动
            current_time = time.time()
            if current_time - last_movement_time > movement_duration:
                command = movement_pattern[pattern_index]
                send_command(command)
                pattern_index = (pattern_index + 1) % len(movement_pattern)
                last_movement_time = current_time
                
            if silver_boxes:
                state = "tracking"
                send_command('S')
                print("Silver ball found! Switching to tracking mode.")
        
        elif state == "tracking":
            if silver_boxes:
                best_box = max(silver_boxes, key=lambda x: x[0])
                ball_distance = best_box[1]
                ball_width = best_box[3]
                
                if abs(ball_distance) > distance_threshold:
                    if ball_distance > 0:
                        send_command('R')
                    else:
                        send_command('L')
                else:
                    if ball_width < proximity_threshold:
                        send_command('F')
                    else:
                        send_command('S')
                        print("Ball in position, initiating grab sequence...")
                        send_command('D')  # Lower servos
                        time.sleep(grab_sequence_delay)
                        send_command('P')  # Raise servos
                        time.sleep(grab_sequence_delay)
                        state = "waiting"
                        wait_start_time = time.time()
                        print("Grab sequence completed, waiting...")
            else:
                state = "exploring"
                print("Lost silver ball. Switching back to exploring mode.")
        
        elif state == "waiting":
            send_command('S')
            remaining_time = int(wait_duration - (time.time() - wait_start_time))
            if remaining_time <= 0:
                state = "exploring"
                print("Wait complete. Switching back to exploring mode.")
            else:
                cv2.putText(cv2_img, f"Waiting: {remaining_time}s", (10, 70), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        
        # 更新FPS计数器
        counter += 1
        if time.perf_counter() - fps_time > 1:
            fps = int(counter / (time.perf_counter() - fps_time))
            fps_time = time.perf_counter()
            counter = 0
        
        # 显示状态信息
        cv2.putText(cv2_img, f"Target: {target_ball}", (10, 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cv2_img, f"State: {state}", (10, 45), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 显示FPS
        cv2.putText(cv2_img, str(fps), text_pos, 
                    cv2.FONT_HERSHEY_DUPLEX, .7, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 显示帧
        cv2.imshow('Zone Camera', cv2_img)
        
        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_command('S')
            break
    
    # 释放资源
    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    zone_cam_loop()
