import open3d as o3d
import numpy as np


def labels_to_colors(labels, num_classes):
    """
    Map 5 segmentation classes to fixed high-contrast RGB colors.
    """

    color_map = np.array([
        [0.00, 0.45, 0.85],  # Class 0 - Blue
        [0.85, 0.10, 0.10],  # Class 1 - Red
        [0.10, 0.70, 0.20],  # Class 2 - Green
        [0.95, 0.85, 0.10],  # Class 3 - Yellow
        [0.60, 0.20, 0.80],  # Class 4 - Purple
    ], dtype=np.float32)

    return color_map[labels]


def visualize_pointcloud(xyz, labels, num_classes, title="PointCloud"):
    """
    Visualize point cloud with colors mapped from labels.
    """
    colors = labels_to_colors(labels, num_classes)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Showing: {title}")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_side_by_side(xyz, labels, preds, num_classes):
    """ 
    A single window comparing GT and predictions 
    """
    colors_gt = labels_to_colors(labels, num_classes)
    colors_pred = labels_to_colors(preds, num_classes)

    # Offset prediction cloud along X axis
    xyz_pred = xyz.copy()
    xyz_pred[:, 0] += 11.0  # shift right

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(xyz)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(xyz_pred)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_pred],
        window_name="GT (left) vs Prediction (right)"
    )


def visualize_errors(xyz, labels, preds):
    """ 
    Highlight misclassified points only 
    """
    errors = labels != preds
    colors = np.zeros((len(labels), 3))
    colors[errors] = [1, 0, 0]      # red = wrong
    colors[~errors] = [0.7, 0.7, 0.7]  # gray = correct

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Errors (Red)")
