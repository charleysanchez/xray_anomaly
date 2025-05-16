import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from evaluate import evaluate_model
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import ChestXrayDataset
from tqdm import tqdm
from utils import train_test_val_splits

def train_model(train_dataset, val_dataset, label_columns, num_epochs=10, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, len(label_columns)),
        nn.Sigmoid()
    )
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        eval_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                eval_images = batch['image'].to(device)
                eval_labels = batch['labels'].to(device)

                outputs = model(eval_images)
                eval_loss += criterion(outputs, eval_labels).item()
                num_batches += 1
        
        eval_loss = eval_loss / num_batches

            
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f} | Eval loss: {eval_loss:.4f}")

        wandb.log(
            {
                "Train Loss": avg_loss,
                "Eval Loss": eval_loss
            }
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "eval_loss": eval_loss,
                "epoch": epoch+1
            },
            f"models/resnet50_epoch_{epoch+1}.pt"
        )

    return model


if __name__ == '__main__':
    wandb.init(project='xray')
    dataset = train_test_val_splits()
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_columns=train_dataset.label_columns,
        num_epochs=10,
        batch_size=16,
        lr=1e-4
    )

