"""
This is a simple implementation of PointNet for point cloud segmentation.
The model uses only XYZ coordinates as the RGB information is tightly coupled with the tags
in the dataset and may not generalize well. 
The input to the model is a point cloud of shape (B, N, 3), where B is the batch size, 
N is the number of points, and 3 corresponds to the XYZ coordinates. 
The output is a per-point class score of shape (B, N, num_classes). 

Simplification
The model used Conv1D layers to replace the shared MLP as mentioned in the orginal paper.
The rotation T-nets are not implemented as the dataset is already aligned and the model can 
learn to be invariant to rotations through data augmentation.
"""
import torch
import torch.nn as nn


# -------------------------------------------------
# Segmentation model (XYZ only input)
# -------------------------------------------------
class PointNetSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.local_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.global_mlp = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Conv1d(64 + 1024, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(1, 2)  # B x 3 x N

        local_feat = self.local_mlp(x)
        global_feat = self.global_mlp(local_feat)

        global_feat = torch.max(global_feat, 2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, N)

        feat = torch.cat([local_feat, global_feat], dim=1)

        out = self.head(feat)
        return out.transpose(1, 2)
