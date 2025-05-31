import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from evaluate import evaluate_model
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import ChestXrayDataset
from model import get_class_balanced_weights
from tqdm import tqdm
from utils import train_test_val_splits

def train_model(train_dataset, val_dataset, label_columns, num_epochs=10, batch_size=32, lr=1e-4):
    wandb.init(project='xray', config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "labels": label_columns
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(label_columns)),
        nn.Sigmoid()
    )
    model = model.to(device)

    criterion = nn.BCELoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} / {num_epochs}"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        
        metrics = evaluate_model(model, val_loader, criterion, device, label_columns)

            
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | Mean AUROC: {metrics['mean_auroc']:.4f}")

        wandb.log(
            {
                "epoch": epoch+1,
                "Train Loss": avg_loss,
                **metrics
            }
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "epoch": epoch+1,
                **metrics
            },
            f"models/resnet50_BCE_epoch_{epoch+1}.pt"
        )

        scheduler.step(metrics['val_loss'])


    return model


if __name__ == '__main__':

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        
    train_transform = transforms.Compose([
        # random crop & scale like CheXNet
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, fill=0),
        # radiograph “texture” augmentations
        transforms.RandomEqualize(p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        # if your X-rays are single-channel, repeat to 3 ch
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
        normalize
    ])

    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
        normalize,
    ])

    dataset = train_test_val_splits(train_transform=train_transform, val_transform=val_transform)
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_columns=train_dataset.label_columns,
        num_epochs=100,
        batch_size=16,
        lr=1e-4
    )

