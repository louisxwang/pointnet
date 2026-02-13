import os
import json
import numpy as np
from pathlib import Path
import torch
from sklearn.neighbors import NearestNeighbors

from model import PointNetSeg


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class SceneEvalDataset:
    def __init__(self, data_dir, split='test'):
        """ Collect data points from all files inside given folder 
        Params
            data_dir: folder containing scene_0x.npz and splits.json
            split: which files to load, choose from 'train', 'val', 'test', 
                cf splits.json
        """
        self.root = Path(data_dir).resolve()

        with open(self.root / "splits.json") as f:
            splits = json.load(f)

        self.scene_ids = splits[split]
        self.files = [self.root / f"scene_0{sid}.npz" for sid in self.scene_ids]

    def __len__(self):
        """ Number of files inside the dataset """
        return len(self.files)

    def __getitem__(self, idx):
        """ Returns all points inside the i-th file """
        data = np.load(self.files[idx])
        xyz = data["xyz"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        return xyz, labels, self.files[idx]


# --------------------------------------------------
# Inference
# --------------------------------------------------
def sliding_window_inference(
    model,
    xyz,
    device,
    num_classes,
    block_size=2.0,
    stride=1,
    num_points=4096,
    num_votes=3,
):
    """
    Inference on given data points using the sliding window technique
    Params
        model: the PointNet model used for inference
        xyz: (N,3) the data points
        device: torch.device instance, determines if GPU is used
        num_classes: how many classes does the model support
        block_size: edge length of the sliding window, unit: same as the dataset
        stride: step amount of the sliding window, unit: same as the dataset 
        num_points: how many points does the model expect for input
        num_votes: number of repeat for the sampling + inference inside 
            each window
    Returns
        final_preds: (N) class label of each point
    """

    model.eval()

    N = xyz.shape[0]
    print(f"n_points {N}")
    vote_logits = np.zeros((N, num_classes))
    vote_counts = np.zeros(N)

    coord_min = np.min(xyz[:, :2], axis=0)
    coord_max = np.max(xyz[:, :2], axis=0)

    x_steps = np.arange(coord_min[0], coord_max[0], stride)
    y_steps = np.arange(coord_min[1], coord_max[1], stride)

    with torch.no_grad():
        for x in x_steps:
            for y in y_steps:
                x_cond = (xyz[:, 0] >= x) & (xyz[:, 0] <= x + block_size)
                y_cond = (xyz[:, 1] >= y) & (xyz[:, 1] <= y + block_size)
                block_mask = x_cond & y_cond
                block_indices = np.where(block_mask)[0]
                n_pts= len(block_indices)
                print(f"x {x}, y {y}, n_pts {n_pts}")
                if n_pts < 1000:
                    continue
                
                n_repeat = max(num_votes, n_pts//num_points + 1)
                for _ in range(n_repeat):
                    choice = np.random.choice(block_indices, num_points, 
                            replace=n_pts < num_points)

                    block_points = xyz[choice]

                    block_tensor = (
                        torch.from_numpy(block_points)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                    )

                    preds = model(block_tensor)  # (1,4096,C)
                    preds = preds.squeeze(0).cpu().numpy()

                    vote_logits[choice] += preds
                    vote_counts[choice] += 1

    # --------------------------------------------------
    # Normalize votes
    # --------------------------------------------------
    nonzero_mask = vote_counts > 0
    vote_logits[nonzero_mask] /= vote_counts[nonzero_mask][:, None]

    final_preds = np.zeros(N, dtype=np.int64)
    final_preds[nonzero_mask] = np.argmax(
        vote_logits[nonzero_mask], axis=1
    )

    # --------------------------------------------------
    # Handle zero-vote points via NN interpolation
    # --------------------------------------------------
    zero_mask = vote_counts == 0

    if np.sum(zero_mask) > 0:
        print(f"Warning: {np.sum(zero_mask)} zero-vote points. Applying NN fill.")

        voted_xyz = xyz[nonzero_mask]
        zero_xyz = xyz[zero_mask]

        nbrs = NearestNeighbors(n_neighbors=1).fit(voted_xyz)
        _, indices = nbrs.kneighbors(zero_xyz)

        final_preds[zero_mask] = final_preds[nonzero_mask][indices[:, 0]]

    return final_preds


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(preds, labels, num_classes):

    overall_acc = (preds == labels).sum() / len(labels)

    class_acc = []
    ious = []

    for cls in range(num_classes):
        pred_mask = preds == cls
        label_mask = labels == cls

        intersection = np.sum(pred_mask & label_mask)
        union = np.sum(pred_mask | label_mask)
        total_label = np.sum(label_mask)

        if total_label > 0:
            class_acc.append(intersection / total_label)

        if union > 0:
            ious.append(intersection / union)

    return (
        overall_acc,
        np.mean(class_acc),
        np.mean(ious),
    )


# --------------------------------------------------
# Evaluation Loop
# --------------------------------------------------
def evaluate(model, dataset, device, num_classes):

    all_preds = []
    all_labels = []

    for i in range(len(dataset)):
        xyz, labels, filepath = dataset[i]
        print(f"Evaluating {os.path.basename(filepath)}")

        preds = sliding_window_inference(
            model,
            xyz,
            device,
            num_classes,
        )

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return compute_metrics(all_preds, all_labels, num_classes)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data/input"
    checkpoint_path = "./artifacts/best_model.pth"

    num_classes = 5 

    model = PointNetSeg(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = SceneEvalDataset(data_dir)

    overall_acc, mean_class_acc, mean_iou = evaluate(
        model,
        dataset,
        device,
        num_classes,
    )

    print("\n===== S3DIS-style Evaluation =====")
    print(f"Overall Accuracy:     {overall_acc:.4f}")
    print(f"Mean Class Accuracy:  {mean_class_acc:.4f}")
    print(f"Mean IoU:             {mean_iou:.4f}")


if __name__ == "__main__":
    main()
