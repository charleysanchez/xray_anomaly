import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

def find_optimal_thresholds(y_true, y_pred, label_columns):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 50):
            pred_bin = (y_pred[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds.append(best_t)
    print("✅ Optimal thresholds per class:")
    for label, t in zip(label_columns, thresholds):
        print(f"  {label}: {t:.2f}")
    return thresholds

def evaluate_model(model, dataloader, criterion, device, label_columns):
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

            all_outputs.append(torch.sigmoid(outputs).cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_outputs).numpy()
    y_true = torch.cat(all_labels).numpy()

    thresholds = find_optimal_thresholds(y_true, y_pred, label_columns)


    per_class_aurocs = []
    per_class_f1s = []

    for i in range(len(label_columns)):
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auroc = float('nan')
        per_class_aurocs.append(auroc)

        preds_bin = (y_pred[:, i] >= thresholds[i]).astype(int)
        f1 = f1_score(y_true[:, i], preds_bin, zero_division=0)
        per_class_f1s.append(f1)

    metrics = {
        "val_loss": val_loss / len(dataloader),
        "mean_auroc": sum([x for x in per_class_aurocs if not np.isnan(x)]) / len(label_columns),
        "mean_f1": sum(per_class_f1s) / len(label_columns),
    }

    # Also include per-class breakdowns in metrics
    for label, auroc, f1 in zip(label_columns, per_class_aurocs, per_class_f1s):
        metrics[f"AUROC_{label}"] = auroc
        metrics[f"F1_{label}"] = f1


    print("\n📊 Summary:")
    print("  Predicted positives per class:", (y_pred >= 0.5).sum(axis=0).astype(int))
    print("  Mean AUROC: {:.4f}, Mean F1 (best threshold): {:.4f}".format(
        metrics["mean_auroc"], metrics["mean_f1"]
    ))
    return metrics

    
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
