"""
This is the main entry point for training and testing the PointNet model.
It parses command line arguments to determine whether to run training or testing,
and then calls the appropriate functions in src/pointnet/train.py and src/pointnet/eval.py
For details on the arguments and their usage, run `python main.py --help`
"""

import argparse

from src.pointnet.eval import evaluate
from src.pointnet.train import train


def get_args():
    parser = argparse.ArgumentParser(
        description="Training/testing configuration for PointNet"
    )
    parser.add_argument("--train", action="store_true",
        help="Run training"
    )
    parser.add_argument("--test", action="store_true",
        help="Run testing"
    )
    parser.add_argument("--data_dir", type=str, default="./data/input",
        help="Path to the input dataset directory"
    )
    parser.add_argument("--num_points", type=int, default=4096,
        help="Model input size (number of points)"
    )
    parser.add_argument("--use_sampling_cube", action="store_true",
        help="Experimental, sample points in a random positioned cube volume"
    )
    parser.add_argument("--artifact_dir", type=str, default="./artifacts",
        help="Train only: Directory to store checkpoints, logs, and outputs"
    )
    parser.add_argument("--batch_size", type=int, default=32,
        help="Train only: Batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=100,
        help="Train only: Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Train only: Learning rate"
    )
    parser.add_argument("--checkpoint_path", type=str, default="./artifacts/best_model.pth",
        help="Test only: Where to load the model for test"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.train:
        print("Training")
        train(
            data_dir=args.data_dir,
            artifact_dir=args.artifact_dir,
            batch_size=args.batch_size,
            num_points=args.num_points,
            epochs=args.epochs,
            lr=args.lr,
            use_sampling_cube=args.use_sampling_cube
            )
    else:
        print("Testing")
        evaluate(
            data_dir =args.data_dir,
            checkpoint_path =args.checkpoint_path,
            use_sampling_cube=args.use_sampling_cube
            )
