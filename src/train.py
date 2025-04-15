import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    all_preds = []
    all_targets = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(labels.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='micro')
    return running_loss / len(dataloader), f1
