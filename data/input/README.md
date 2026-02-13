# Synthetic 3D Segmentation Dataset (Point Clouds)

Scenes are composed of simple primitives:
- ground plane (class 1)
- box (class 2)
- sphere (class 3)
- cylinder (class 4)

Per-scene files:
- `scene_XX.npz`: `xyz` (Nx3 float32), `rgb` (Nx3 float32), `labels` (N int64)
- `scene_XX.ply`: visualization helper (XYZRGB), **no labels** in PLY

Metadata:
- `classes.json`: class id -> name
- `splits.json`: train/val/test split by scene id

Suggested evaluation:
- per-class IoU and mean IoU on validation scenes.
- qualitative visualizations by coloring points by predicted label.
