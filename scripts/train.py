import torch
from models.yowov3 import YOWOv3
from utils.data_utils import load_video_clip
from utils.train_utils import compute_loss

#TODO Change with actual data loading
model = YOWOv3(num_classes=24, num_anchors=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    #TODO Replace with dataloader
    clip = torch.tensor(load_video_clip('path_to_video.mp4'), 
                        dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)
    gt_boxes = torch,tensor([])
    gt_action = torch.tensor([])
    pred_boxes, pred_actions = model(clip)
    loss = compute_loss(pred_boxes, pred_actions, gt_boxes, gt_action) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 
    
    #TODO Save model checkpoint