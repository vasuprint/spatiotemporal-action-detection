import cv2
import numpy as np

def load_video_clip(path, num_frame=16):
    cap = cv2.VideoCapture(path)
    frames = []
    
    for _ in range(num_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame =cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    return np.array(frames)