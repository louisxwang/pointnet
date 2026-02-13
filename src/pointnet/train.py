import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import PointNetSeg


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class SceneDataset(Dataset):
    def __init__(self, root, split, num_points=4096, samples_per_scene=64, 
                 normalize=False, use_sampling_cube=False):
        self.root = Path(root).resolve()
        self.num_points = num_points
        self.use_sampling_cube = use_sampling_cube  # sample inside a random cube region

        with open(self.root / "splits.json") as f:
            splits = json.load(f)

        self.scene_ids = splits[split]
        self.files = [self.root / f"scene_0{sid}.npz" for sid in self.scene_ids]

        if normalize:
            self.scenes = []
            for f in self.files:
                data = np.load(f)
                xyz = data["xyz"] # (N,3)
                print(f"data mean {np.mean(xyz, 0)}")
                print(f"data min max {np.min(xyz, axis=0)}, {np.max(xyz, axis=0)}")
                xyz = xyz - xyz.mean(0)
                xyz = xyz / (np.max(np.linalg.norm(xyz, axis=1))+ 1e-6)
                self.scenes.append({"xyz":xyz, "labels":data["labels"]})
                self.sampling_cube_size = 0.5  # Random sampling from a cube volume
        else: # Normalization desactived as all files are within [-5, 5] for x,y and [0, 2] for z
            self.scenes = [np.load(f) for f in self.files]
            self.sampling_cube_size = 2

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

        # Random sampling from a cube volume
        block_points = [] if self.use_sampling_cube else xyz
        block_labels = [] if self.use_sampling_cube else labels
        while len(block_points) < self.num_points:
            # Pick random point as cube center
            center_idx = np.random.choice(len(xyz))
            center = xyz[center_idx]

            mask = (
                (xyz[:, 0] > center[0] - self.sampling_cube_size/2) &
                (xyz[:, 0] < center[0] + self.sampling_cube_size/2) &
                (xyz[:, 1] > center[1] - self.sampling_cube_size/2) &
                (xyz[:, 1] < center[1] + self.sampling_cube_size/2)
            )

            block_points = xyz[mask] - center
            block_labels = labels[mask]

        choice = np.random.choice(
            len(block_points),
            self.num_points,
            replace=False
        )

        block_points = block_points[choice]
        block_labels = block_labels[choice]

        # Random Z rotation, this avoids overfitting
        theta = np.random.uniform(0, 2 * np.pi)
        cosval = np.cos(theta)
        sinval = np.sin(theta)

        rotation_matrix = np.array([
            [cosval, -sinval, 0],
            [sinval,  cosval, 0],
            [0,       0,      1]
        ])

        block_points = block_points @ rotation_matrix  
        
        return (
            torch.tensor(block_points, dtype=torch.float32),
            torch.tensor(block_labels, dtype=torch.long),
        )


# -------------------------------------------------
# Training
# -------------------------------------------------

def train(
    data_dir = "./data/input",
    artifact_dir = "./artifacts",
    batch_size = 32,
    num_points = 4096,
    epochs = 100,
    lr = 1e-3,
    use_sampling_cube=False
):

    os.makedirs(artifact_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Classes
    with open(os.path.join(data_dir, "classes.json")) as f:
        classes = json.load(f)
    print(f"Classes {classes}")

    num_classes = len(classes)

    # Datasets
    train_set = SceneDataset(data_dir, "train", num_points, use_sampling_cube=use_sampling_cube)
    val_set = SceneDataset(data_dir, "val", num_points, use_sampling_cube=use_sampling_cube)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Model
    model = PointNetSeg(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
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

        # ---- Checkpoint Saving ----
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

        # Save last checkpoint every epoch
        torch.save(checkpoint, os.path.join(artifact_dir, "last_checkpoint.pth"))

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(artifact_dir, "best_model.pth"))
            print("Best model saved.")
