from ultralytics import YOLO
import cv2
import numpy as np

model=YOLO('yolov8n.pt')

cap= cv2.VideoCapture(0)

if not cap.isOpened:
    print("Error:could not open webcame")
    exit()
else:
    while True:
       ret, frame = cap.read()
       if not ret:
            print('Failde to grab frame')
            break
       results=model(frame)
       annotated_frame=results[0].plot()

       cv2.imshow("YOLO live detection",annotated_frame)

       if cv2.waitKey(1)== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
       
      