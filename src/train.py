import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import ChestXrayDataset
from tqdm import tqdm
from utils import train_test_val_splits

def train_model(train_dataset, val_dataset, label_columns, num_epochs=10, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.resnet50(pretrained=True)
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
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f}")

    return model


if __name__ == '__main__':
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