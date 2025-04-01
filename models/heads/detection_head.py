import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(DetectionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
        
        