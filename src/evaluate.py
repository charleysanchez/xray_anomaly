import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

def find_optimal_thresholds(y_true, y_pred, label_columns):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.0, 0.0
        for t in np.linspace(0.05, 0.95, 91):
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
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 1) Get the 5D input and 2D labels
            images = batch["image"]   # [B, 10, 3, 224, 224], on CPU
            labels = batch["labels"]  # [B, num_labels], on CPU

            # Move labels to GPU
            labels = labels.to(device)

            # 2) Flatten crop dimension BEFORE sending to GPU
            bs, ncrops, c, h, w = images.size()
            images = images.view(bs * ncrops, c, h, w)   # [B*10, 3, 224, 224] on CPU
            images = images.to(device)                   # now on GPU

            # Forward pass on all crops at once:
            outputs_flat = model(images)                   # [B*10, num_labels]

            # Reshape so that outputs_flat → [B, 10, num_labels]:
            _, L = outputs_flat.size()
            outputs = outputs_flat.view(bs, ncrops, L)         # [B, 10, L]

            # Now collapse the 10 predictions per image (e.g. by averaging):
            outputs = outputs.mean(dim=1)                       # [B, L]

            # Compute loss against the true labels (no cropping here):
            loss = criterion(outputs, labels)
            all_losses.append(loss.item())

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    val_loss = np.mean(all_losses)

    y_pred_probs = torch.cat(all_outputs).numpy()
    y_true = torch.cat(all_labels).numpy()

    thresholds = find_optimal_thresholds(y_true, y_pred_probs, label_columns)

    # Binarize predictions using optimal thresholds for each class
    y_pred_binary_optimal = np.zeros_like(y_pred_probs)
    for i in range(len(label_columns)):
        y_pred_binary_optimal[:, i] = (y_pred_probs[:, i] >= thresholds[i]).astype(int)


    per_class_aurocs = []
    per_class_f1s = []

    for i in range(len(label_columns)):
        # AUROC
        try:
            # Ensure y_true has both classes for this label if it's not all 0s or all 1s
            if len(np.unique(y_true[:, i])) > 1:
                auroc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
            else: # Only one class present in y_true, AUROC is undefined or 0.5 by convention
                auroc = float('nan') # Or 0.5, depending on desired behavior
        except ValueError:
            auroc = float('nan') # Should be caught by len(np.unique()) check too
        per_class_aurocs.append(auroc)

        # F1 Score (using optimal threshold for this class)
        # y_pred_binary_optimal[:, i] is already thresholded
        f1 = f1_score(y_true[:, i], y_pred_binary_optimal[:, i], zero_division=0)
        per_class_f1s.append(f1)

    # Calculate new metric: Overall Label Prediction Accuracy
    correct_label_predictions = (y_pred_binary_optimal == y_true).sum()
    total_label_predictions = y_true.size
    overall_label_accuracy = (correct_label_predictions / total_label_predictions) * 100

    metrics = {
        "val_loss": val_loss, #/ len(dataloader),
        "mean_auroc": np.nanmean(per_class_aurocs), # Use nanmean to correctly handle NaNs
        "mean_f1_optimal_threshold": sum(per_class_f1s) / len(label_columns), # This is Macro F1
        "overall_label_accuracy_percentage": overall_label_accuracy,
    }

    # Also include per-class breakdowns in metrics
    for i, label in enumerate(label_columns):
        metrics[f"AUROC_{label}"] = per_class_aurocs[i]
        metrics[f"F1_optimal_threshold_{label}"] = per_class_f1s[i]


    print("\n📊 Summary:")
    predicted_positives_optimal = y_pred_binary_optimal.sum(axis=0).astype(int)
    print("  Predicted positives per class (using optimal thresholds):")
    for i, label in enumerate(label_columns):
        print(f"    {label}: {predicted_positives_optimal[i]}")

    print(f"  Mean AUROC: {metrics['mean_auroc']:.4f}")
    print(f"  Mean F1 (optimal thresholds): {metrics['mean_f1_optimal_threshold']:.4f}")
    print(f"  Overall Label Prediction Accuracy (optimal thresholds): {metrics['overall_label_accuracy_percentage']:.2f}%") # New metric
    
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
