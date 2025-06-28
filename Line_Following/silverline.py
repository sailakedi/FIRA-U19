from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('silver_classify_s.pt')  # Path to your trained model

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    # Read frame from camera
    success, frame = cap.read()
    
    if not success:
        print("Error reading frame")
        break

    # Run YOLO inference
    results = model(frame)  # Pass frame directly to the model
    
    # Visualize results on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()