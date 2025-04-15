import torch
from sklearn.metrics import f1_score

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(labels.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='micro')
    return val_loss / len(dataloader), f1
