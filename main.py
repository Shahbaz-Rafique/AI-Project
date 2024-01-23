import cv2
import torch
import numpy as np

path = 'best.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

video_path = 'movie2.mp4'
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    frame = np.squeeze(results.render())

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

