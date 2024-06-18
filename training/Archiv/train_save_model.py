
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from training.load_data import  train_loader, val_loader
from model import Predictor


def train_model(num_classes, train_loader, val_loader):
    model = Predictor()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0000001)

    best_accuracy = 0.0
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        #Training Loop
        for b_id, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            predictions = model(images)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #Validation Loop
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
  

                predicted = torch.round(outputs).squeeze()
                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()
  
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
        torch.save(model.state_dict(),r'C:\Users\nme.AD\Desktop\FestoCode\best_model.pth')

    print('Finished Training')

if __name__ == '__main__':
    data_dir = r"C:\Users\nme.AD\Desktop\Festo_Dataset\all_pictures"

    #train_loader = dataloader #, val_loader, num_classes
    train_model(2, train_loader, val_loader)
