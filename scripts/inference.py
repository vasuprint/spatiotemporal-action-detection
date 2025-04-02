import torch
from models.yowov3 import YOWOv3
from utils.data_utils import load_video_clip
from utils.vis_utils import draw_boxes

model = YOWOv3(num_classes=80, num_anchors=3)

# Load the model weights
# model.load_state_dict(torch.load('path_to_weights.pth'))
model.eval()

clip = torch.tensor(load_video_clip('path_to_video.mp4'),dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
with torch.no_grad():
    boxes, action = model(clip)
    
# Visualize the results
frame = clip[0, :, 0, :, :].permute(1, 2, 0).numpy()
result = draw_boxes(frame, pred_boxes[0], pred_actions[0])
cv2.imshow('Result', result)    
cv2.waitKey(0)