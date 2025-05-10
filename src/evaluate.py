import os
import torch
from sklearn.metrics import f1_score, roc_auc_score

def validate_model(model, val_loader, criterion, device, label_columns):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    y_pred = all_preds.numpy()
    y_true = all_targets.numpy()

    aurocs = []
    f1s = []

    for i in range(len(label_columns)):
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auroc = float('nan')
        aurocs.append(auroc)

        preds_bin = (y_pred[:, i] >= 0.5).astype(int)
        f1 = f1_score(y_true[:, i], preds_bin, zero_division=0)
        f1s.append(f1)

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, aurocs, f1s
    
def save_best_model(model, optimizer, epoch, val_loss, best_val_loss, save_path='checkpoints/best_model.pt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if val_loss < best_val_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, save_path)
        print(f"✅ Best model saved at epoch {epoch} with val loss {val_loss:.4f}")
        return val_loss
    return best_val_loss
