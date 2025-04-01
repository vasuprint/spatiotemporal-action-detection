import torch
import torch.nn as nn

class I3DBackbone(nn.Module):
    def __init__(self):
        super(I3DBackbone, self).__init__()
        self.i3d = torch.hub.load('pytorch/vision', 'i3d_r50',
                                    pretrained=True)
        self.i3d = nn.sequential(*list(self.i3d.children())[:-2]) # remove the last two layers
        
    def forward(self, x):
        return self.i3d(x)
    