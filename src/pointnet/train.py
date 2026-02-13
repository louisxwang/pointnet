import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# Model (RGB only input)
# Small dataset, full model would overfit
# Simplification are made by using less layers
# -------------------------------------------------

def block(in_c, out_c):
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, 1),
        nn.BatchNorm1d(out_c),
        nn.ReLU()
    )


class PointNetSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.local_mlp = block(3, 64)
        self.global_mlp = block(64, 1024)

        self.head = nn.Sequential(
            block(64 + 1024, 256),
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


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class SceneDataset(Dataset):
    def __init__(self, root, split, num_points=4096, samples_per_scene=100):
        self.root = Path(root).resolve()
        self.num_points = num_points

        with open(self.root / "splits.json") as f:
            splits = json.load(f)

        self.scene_ids = splits[split]
        self.files = [self.root / f"scene_0{sid}.npz" for sid in self.scene_ids]
        self.scenes = [np.load(f) for f in self.files]

        # Multiple sampels from each file
        self.index_map = []
        for i in range(len(self.files)):
            self.index_map.extend([i]*samples_per_scene)
                       
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        scene_idx = self.index_map[idx]
        data = self.scenes[scene_idx]

        xyz = data["xyz"]          # (N,3)
        labels = data["labels"]    # (N,)

        # Normalize xyz (center + scale)
        xyz = xyz - xyz.mean(0)
        scale = np.max(np.linalg.norm(xyz, axis=1))
        xyz = xyz / (scale + 1e-6)

        # Random sampling
        N = xyz.shape[0]
        choice = np.random.choice(N, self.num_points, replace=N<self.num_points)

        xyz = xyz[choice]
        labels = labels[choice]

        points = xyz + np.random.normal(0, 0.01, xyz.shape) 

        return (
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )


# -------------------------------------------------
# Training
# -------------------------------------------------

def train():
    root = "./data/input"
    batch_size = 16
    num_points = 4096
    epochs = 100
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Classes
    with open(os.path.join(root, "classes.json")) as f:
        classes = json.load(f)
    print(f"Classes {classes}")

    num_classes = len(classes)

    # Datasets
    train_set = SceneDataset(root, "train", num_points)
    val_set = SceneDataset(root, "val", num_points)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model
    model = PointNetSeg(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_points = 0

        for points, labels in tqdm(train_loader):
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(points)

            loss = criterion(
                preds.reshape(-1, num_classes),
                labels.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred_labels = preds.argmax(dim=2)
            # print(pred_labels)
            # print(labels)
            total_correct += (pred_labels == labels).sum().item()
            total_points += labels.numel()

        train_acc = total_correct / total_points

        # Validation
        model.eval()
        val_correct = 0
        val_points = 0

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)

                preds = model(points)
                pred_labels = preds.argmax(dim=2)

                val_correct += (pred_labels == labels).sum().item()
                val_points += labels.numel()

        val_acc = val_correct / val_points

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss {total_loss/len(train_loader):.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )


if __name__ == "__main__":
    train()
