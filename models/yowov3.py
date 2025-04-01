import torch
import torch.nn as nn
from models.backbones.yolo_backbone import YOLOBackbone
from models.backbones.i3d_backbone import I3DBackbone
from models.heads.detection_head import DetectionHead
from models.heads.classification_head import ClassificationHead

class YOWOv3(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(YOWOv3, self).__init__()
        self.backbone2D =YOLOBackbone()
        self.backbone3D = I3DBackbone()
        
        # TODO Fix Fusion Layer (adjust channels later)
        self.fusion_layer = nn.Conv2d(1024, 512, kernel_size=1) # TODO Adjust based on actual dim
        self.detection_head = DetectionHead(512, num_anchors, num_classes)
        self.classification_head = ClassificationHead(512, num_classes)
        
        def forward(self, x):
            feat_2d = [self.backbone2D(x[:, :, t, :, :]) for t in range(x.shape[2])]
            feat_2d = torch.stack(feat_2d, dim=2)
            feat_3d = self.backbone3D(x)
            
            # TODO Simplify Fusion Layer
            combined = torch.cat([feat_2d.mean(dim=2), feat_3d.squeeze(2)],
                                 dim=1)
            combined - self.fusion_layer(combined)
            boxes = self.detection_head(combined)
            action = self.classification_head(combined.mean(dim=(2,3)))
            return boxes, action