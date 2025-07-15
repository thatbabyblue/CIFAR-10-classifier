import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.jit.log_extract import run_nnc

from model import myCNN
from dataloader import get_dataloaders

def train_model():
    train_loader, _ = get_dataloaders()
    model = myCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "cifar10_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_model()