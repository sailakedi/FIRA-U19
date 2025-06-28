import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import torch

# Camera settings
camera_width = 640
camera_height = 480
horizontal_center = camera_width // 2
text_pos = np.array([int(camera_width * 0.96), int(camera_height * 0.04)])

# Initialize variables for ball tracking
ball_distance = 0
ball_type = "none"
ball_width = -1
last_best_box = None

def save_image(image):
    if not os.path.exists("datasets/images_to_annotate/zone_images"):
        os.makedirs("datasets/images_to_annotate/zone_images")
    num = len(os.listdir("datasets/images_to_annotate/zone_images"))
    cv2.imwrite(f"datasets/images_to_annotate/zone_images/{num:04d}.png", image)

def zone_cam_loop():
    global ball_distance, ball_type, ball_width, last_best_box

    # Initialize FPS counter variables
    fps_time = time.perf_counter()
    counter = 0
    fps = 0

    # Load YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('ball_detect_s.pt')
    model.to(device)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    crop_percentage = 0.45
    crop_height = int(camera_height * crop_percentage)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Crop the image
        cv2_img = frame[crop_height:, :]

        # Run YOLO detection
        results = model.predict(cv2_img, 
                              imgsz=(512, 224), 
                              conf=0.3, 
                              iou=0.2, 
                              agnostic_nms=True, 
                              verbose=False)

        result = results[0].numpy()
        boxes = []

        # Process detection results
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            class_id = box.cls[0].astype(int)
            name = result.names[class_id]
            confidence = box.conf[0].astype(float)

            width = x2 - x1
            height = y2 - y1
            area = width * height
            distance = (x1 + width // 2) - horizontal_center
            boxes.append([area, distance, name, width])

            color = colors(class_id, True)
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv2_img, f"{name}: {confidence:.2f}", 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_DUPLEX, 
                       0.5, 
                       color, 
                       1, 
                       cv2.LINE_AA)

        # Update ball tracking information
        if len(boxes) > 0:
            best_box = max(boxes, key=lambda x: x[0])
            if last_best_box is not None:
                best_box = min(boxes, key=lambda x: abs(x[1] - last_best_box[1]))

            last_best_box = best_box
            ball_distance = best_box[1]
            ball_type = str.lower(str(best_box[2]))
            ball_width = best_box[3]
        else:
            last_best_box = None
            ball_distance = 0
            ball_type = "none"
            ball_width = -1

        # Update FPS counter
        counter += 1
        if time.perf_counter() - fps_time > 1:
            fps = int(counter / (time.perf_counter() - fps_time))
            fps_time = time.perf_counter()
            counter = 0

        # Draw FPS
        cv2.putText(cv2_img, 
                    str(fps), 
                    text_pos, 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    .7, 
                    (0, 255, 0), 
                    1, 
                    cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Zone Camera', cv2_img)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    zone_cam_loop()