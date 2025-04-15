# infer.py

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.model import get_model
from src.config import IMG_SIZE, DEVICE, MODEL_NAME


def load_model(checkpoint_path, device):
    model = get_model(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def get_test_images(folder):
    return [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith('.png') or fname.endswith('.jpg')]

def predict(model, image_paths, device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    predictions = []

    for path in tqdm(image_paths, desc="Predicting"):
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.sigmoid(output).item()
            predictions.append((os.path.basename(path), 1 if pred > 0.5 else 0))

    return predictions

def save_predictions(predictions, save_path="test_predictions.csv"):
    with open(save_path, "w") as f:
        f.write("filename,prediction\n")
        for filename, pred in predictions:
            f.write(f"{filename},{pred}\n")
