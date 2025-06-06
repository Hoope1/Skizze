import torch
import torch.nn as nn
class FClipNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,1,1)
    def forward(self,x):
        # placeholder that returns no detections
        return torch.zeros((0,4), device=x.device)
