import torch
import torch.nn as nn


def block(in_c, out_c):
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, 1),
        nn.BatchNorm1d(out_c),
        nn.ReLU()
    )


class PointNetSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Feature extractor (merged)
        self.local_mlp = block(3, 64)       # local feature
        self.global_mlp = block(64, 1024)   # global feature

        # Segmentation head
        self.head = nn.Sequential(
            block(64 + 1024, 256),
            nn.Conv1d(256, num_classes, 1)
        )

    def forward(self, x):
        # x: B x N x 3
        B, N, _ = x.shape
        x = x.transpose(1, 2)  # B x 3 x N

        local_feat = self.local_mlp(x)      # B x 64 x N
        global_feat = self.global_mlp(local_feat)  # B x 1024 x N

        # Global pooling
        global_feat = torch.max(global_feat, 2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, N)

        # Concatenate
        feat = torch.cat([local_feat, global_feat], dim=1)

        out = self.head(feat)
        return out.transpose(1, 2)
