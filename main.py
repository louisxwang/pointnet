from pointnet.train import train
from pointnet.eval import evaluate
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Training/testing configuration for PointNet"
    )
    parser.add_argument("--train", type=bool, action="store_true",
        help="The script will do training"
    )
    parser.add_argument("--test", type=bool, action="store_true",
        help="The script will do testing"
    )
    parser.add_argument("--data_dir", type=str, default="./data/input",
        help="Path to the input dataset directory"
    )
    parser.add_argument("--artifact_dir", type=str, default="./artifacts",
        help="Train only: Directory to store checkpoints, logs, and outputs"
    )
    parser.add_argument("--batch_size", type=int, default=32,
        help="Train only: Batch size for training"
    )
    parser.add_argument("--num_points", type=int, default=4096,
        help="Model input size (number of points)"
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
    parser.add_argument("--num_classes", type=int, default=5,
        help="Test only: Number of classes supported by the model"
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
            num_classes =args.num_classes,
            use_sampling_cube=args.use_sampling_cube
            )

