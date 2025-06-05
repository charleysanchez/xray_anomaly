import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import models, transforms
from dataset import ChestXrayDataset
from utils import train_test_val_splits
import json

# -----------------------------
# STEP A: Load your best model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 15

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
)
checkpoint = torch.load('models/resnet50_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# -----------------------------
# STEP B: Prepare validation DataLoader
# -----------------------------
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
    normalize,
])

# train_test_val_splits returns a dict with keys 'train','val','test'
splits = train_test_val_splits(train_transform=None, val_transform=val_transform)
val_dataset = splits['val']
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# -----------------------------
# STEP C: Gather predictions + labels
# -----------------------------
all_probs = []   # will be shape (N,15)
all_labels = []  # will be shape (N,15)

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['labels'].cpu().numpy()  # shape (B,15)
        logits = model(images)                  # shape (B,15)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

all_probs = np.vstack(all_probs)   # (N,15)
all_labels = np.vstack(all_labels) # (N,15)

# -----------------------------
# STEP D: For each class, find threshold maximizing Youden’s J
# -----------------------------
thresholds = np.arange(0.0, 1.001, 0.01)
best_thresholds = np.zeros(NUM_CLASSES, dtype=float)

for c in range(NUM_CLASSES):
    true_c = all_labels[:, c]  # shape (N,)
    prob_c = all_probs[:, c]   # shape (N,)

    best_j = -1.0
    best_t = 0.5

    for t in thresholds:
        pred_c = (prob_c >= t).astype(int)
        # Build confusion matrix for this class
        # tn, fp, fn, tp = confusion_matrix(true_c, pred_c, labels=[0,1]).ravel()
        # sklearn’s confusion_matrix with labels=[0,1] returns a 2×2 array:
        tn, fp, fn, tp = confusion_matrix(true_c, pred_c, labels=[0,1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1

        if j > best_j:
            best_j = j
            best_t = t

    best_thresholds[c] = best_t

    # (Optional) Compute F1 at that threshold, just to see how it changed
    preds_at_best = (prob_c >= best_t).astype(int)
    f1_at_best = f1_score(true_c, preds_at_best, zero_division=0)

    print(f"Class {c:02d} ({val_dataset.label_columns[c]}): "
          f"J = {best_j:.3f}, best_thresh = {best_t:.2f}, F1@thresh = {f1_at_best:.3f}")

# Save to disk
with open('per_class_thresholds.json', 'w') as f:
    json.dump(
        {val_dataset.label_columns[i]: float(best_thresholds[i]) 
         for i in range(NUM_CLASSES)},
        f, indent=2
    )

print("Saved per-class thresholds to per_class_thresholds.json")
