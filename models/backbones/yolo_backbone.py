import torch.nn as nn
from ultralytics import YOLO

class YOLOBackbone(nn.Module):
    def __init__(self, model_name='yolo11n.pt'):  
        super(YOLOBackbone, self).__init__() 
        yolo_model = YOLO(model_name)
        self.backbone = yolo_model.model.backbone
        
    def forward(self, x):
        return self.backbone(x)
    