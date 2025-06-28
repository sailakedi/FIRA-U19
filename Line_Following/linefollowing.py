import sys
import numpy as np
import cv2

window_title = "USB Camera"

def show_camera():
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video0"
    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
 
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            # Window
            while True:
                ret_val, frame = video_capture.read()
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:

                    height, width = frame.shape[:2]
                    roi = frame[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)]
                    
                    # Convert to grayscale
                    blur = cv2.GaussianBlur(roi,(5,5),0)
                    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                    low_green = np.array([50,100,50])
                    up_green = np.array([70,255,255])
                    green_mask = cv2.inRange(hsv,low_green,up_green)
                    roi[green_mask != 0] = [0, 255, 0]
                    edges_1 = cv2.Canny(green_mask,75,150)
                    lines_1 = cv2.HoughLinesP(edges_1,1,np.pi/180,50,maxLineGap=30)
                    if lines_1 is not None:
                        for line_1 in lines_1:
                            x1,y1,x2,y2 = line_1[0]
                            cv2.line(roi,(x1,y1),(x2,y2),(0,255,0),2)
                            
                    low_black = np.array([0,0,0])
                    up_black = np.array([180,255,50])
                    black_mask = cv2.inRange(hsv,low_black,up_black)
                    roi[black_mask != 0] = [0, 0, 255]  # Red color for the black mask

                    

                    #Find contours and calculate the center of the largest contour
                    contours, hierarchy = cv2.findContours(black_mask, 1, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0: 
                        c = max(contours, key=cv2.contourArea)
                        cv2.polylines(roi, [c], isClosed=True, color=(255,0 ,0), thickness=2)
                        

                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M['m10']/M["m00"])
                            cy = int(M['m10']/M["m00"])
                            print("CX : "+str(cx)+" CY : "+str(cy))
                     

                            

                            # Draw the center (dot) as a blue circle
                            cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)  # Blue dot inside the square
                    
                   




                    # Show the frame
                    
                    cv2.imshow("frame", frame)
                    cv2.imshow(window_title, black_mask)

                else:
                    break
                
                keyCode = cv2.waitKey(10) & 0xFF
                
                if keyCode == 27 or keyCode == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    show_camera()
