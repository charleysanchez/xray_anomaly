import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from evaluate import evaluate_model
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from dataset import ChestXrayDataset
from model import get_class_balanced_weights, DenseNet121
from tqdm import tqdm
from utils import train_test_val_splits

def train_model(train_dataset, val_dataset, label_columns, num_epochs=10, batch_size=32, lr=1e-4, early_stop_patience=5):
    wandb.init(
        project="xray",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "labels": label_columns,
        }
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------
    # DataLoaders (sample no finding at 30% frequency since overwhelming majority of label)
    # ----------------------------
    no_finding_col = train_dataset.label_columns.index("No Finding")
    label_list = []
    for fname in train_dataset.image_names:
        lab = train_dataset.label_dict.get(
        fname,
        np.zeros(NUM_CLASSES, dtype=np.float32)
        )
        label_list.append(lab)
    labels_matrix = np.stack(label_list, axis=0)  # shape [N_train, num_labels]

    is_no_finding = (labels_matrix[:, no_finding_col] == 1)
    weights = np.ones(len(train_dataset), dtype=np.float32)
    weights[is_no_finding] = 0.3   # under‐sample "No Finding" to 30% weight

    sample_weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # ----------------------------
    # Build Model backbone
    # ----------------------------
    model = DenseNet121()
    model = model.to(device)

    # ----------------------------
    # Criterion + Optimizer + Scheduler
    # ----------------------------
    # Class-balanced weights: a tensor of shape [num_labels]
    # class_weights = get_class_balanced_weights(train_dataset).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    criterion = nn.BCELoss(reduction='mean')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images)               # raw logits: [batch_size, num_labels]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        metrics = evaluate_model(model, val_loader, criterion, device, label_columns)
        val_loss = metrics["val_loss"]
        mean_auroc = metrics["mean_auroc"]
        mean_f1   = metrics["mean_f1_optimal_threshold"]

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Mean AUROC: {mean_auroc:.4f} | "
            f"Mean F1: {mean_f1:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": val_loss,
            "Mean AUROC": mean_auroc,
            "Mean F1": mean_f1,
            **{f"AUROC_{lbl}": metrics[f"AUROC_{lbl}"] for lbl in label_columns},
            **{f"F1_{lbl}":    metrics[f"F1_optimal_threshold_{lbl}"] for lbl in label_columns},
            "Overall Label Accuracy": metrics["overall_label_accuracy_percentage"],
        })

        # ----------------------------
        # Check for improvement & Early Stopping
        # ----------------------------
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save only the best checkpoint so far
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "epoch": epoch + 1,
                    **metrics
                },
                "models/densenet121_best.pt"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"→ Early stopping at epoch {epoch+1}")
                break

        # Step the LR scheduler
        scheduler.step(val_loss)

    # At the end, load the best model back before returning
    checkpoint = torch.load("models/densenet121_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model


if __name__ == '__main__':
    NUM_CLASSES = 15

    # Standard ImageNet‐style normalization (even though these are X‐rays, 
    # we repeat them to 3 channels and normalize to ResNet‐pretrained stats)
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # -------------------------------
    # TRAINING TRANSFORMS (SOTA)
    # -------------------------------
    train_transform = transforms.Compose([
        # 1) Random crop to 224×224.
        transforms.RandomResizedCrop(224),

        # 2) Horizontal flip with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),

        # 7) Convert to Tensor
        transforms.ToTensor(),

        # 10) Final normalize to ImageNet statistics
        normalize,
    ])

    # -------------------------------
    # VALIDATION / TEST TRANSFORMS
    # -------------------------------
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(
            lambda crops: torch.stack([
                normalize(transforms.ToTensor()(crop))
                for crop in crops
            ])
        ),
    ])

    # ----------------------------------
    # LOAD / SPLIT DATASET
    # ----------------------------------
    dataset = train_test_val_splits(
        train_transform=train_transform, 
        val_transform=val_transform
    )
    train_dataset = dataset['train']
    val_dataset   = dataset['val']
    test_dataset  = dataset['test']

    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_columns=train_dataset.label_columns,
        num_epochs=50,
        batch_size=16,
        lr=1e-4,
        early_stop_patience=50
    )
