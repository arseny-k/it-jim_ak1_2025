import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torchvision import transforms
import argparse

from src.config import *
from src.dataset import ArtifactDataset
from src.model import get_model
from src.train import train_one_epoch
from src.validate import validate
from src.infer import load_model, get_test_images, predict, save_predictions


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_image_paths_and_labels(folder):
    image_paths, labels = [], []
    for fname in os.listdir(folder):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            label = int(fname.split("_")[-1].split(".")[0])
            image_paths.append(os.path.join(folder, fname))
            labels.append(label)
    return image_paths, labels


def train_model(data_dir, epochs, use_augment=False):
    seed_everything()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    image_paths, labels = get_image_paths_and_labels(data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=SEED
    )


    if use_augment:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        print("Using augmented training transforms.")
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        print("Using basic training transforms (no augmentation).")

    train_ds = ArtifactDataset(train_paths, train_labels, transform)
    val_ds = ArtifactDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_model(MODEL_NAME).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✅ Saved best model!")


def run_inference(data_dir, output_file):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model("models/best_model.pth", device)
    image_paths = get_test_images(data_dir)
    preds = predict(model, image_paths, device)
    save_predictions(preds, output_file)

    y_true = []
    y_pred = []

    artifact_correct = 0
    artifact_total = 0

    for filename, pred in preds:
        true_label = int(filename.split("_")[-1].split(".")[0])
        y_true.append(true_label)
        y_pred.append(pred)

        if true_label == 0:
            artifact_total += 1
            if pred == 0:
                artifact_correct += 1

    f1 = f1_score(y_true, y_pred, average='micro')
    print(f"\n📊 Micro F1 score on test set: {f1:.4f}")

    if artifact_total > 0:
        accuracy = 100 * artifact_correct / artifact_total
        print(f"✅ Artifact-only accuracy: {accuracy:.2f}% ({artifact_correct}/{artifact_total})")
    else:
        print("⚠️ No artifact images found in test set.")


def main():
    parser = argparse.ArgumentParser(description="Train or test the model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train command ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data-dir", type=str, default="./data/train",
        help="Path to the training dataset"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    train_parser.add_argument(
    "--augment", action="store_true",
    help="Enable data augmentation (flip, rotate, jitter)"
    )

    # --- Test command ---
    test_parser = subparsers.add_parser("test", help="Run inference on test set")
    test_parser.add_argument(
        "--data-dir", type=str, default="./data/test",
        help="Path to the test dataset"
    )
    test_parser.add_argument(
        "--output", type=str, default="test_predictions.csv",
        help="CSV file to save predictions"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(data_dir=args.data_dir, epochs=args.epochs, use_augment=args.augment)
    elif args.command == "test":
        run_inference(args.data_dir, args.output)


if __name__ == "__main__":
    main()
